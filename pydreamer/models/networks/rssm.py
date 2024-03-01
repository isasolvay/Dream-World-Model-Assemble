
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import rssm_component
from .. math_functions import *
from .common import *
from .. import tools_v3


class RSSMCore(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm,tidy):
        super().__init__()
        self.cell = RSSMCell(embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm,tidy)

    def forward(self,
                embed: Tensor,       # tensor(T, B, E)
                action: Tensor,      # tensor(T, B, A)
                reset: Tensor,       # tensor(T, B)
                in_state: Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                iwae_samples: int = 1,
                do_open_loop=False,
                ):

        T, B = embed.shape[:2]
        I = iwae_samples

        # Multiply batch dimension by I samples

        def expand(x):
            # (T,B,X) -> (T,BI,X)
            return x.unsqueeze(2).expand(T, B, I, -1).reshape(T, B * I, -1)

        embeds = expand(embed).unbind(0)     # (T,B,...) => List[(BI,...)]
        actions = expand(action).unbind(0)
        reset_masks = expand(~reset.unsqueeze(2)).unbind(0)

        priors = []
        posts = []
        states_h = []
        samples = []
        (h, z) = in_state

        for i in range(T):
            if not do_open_loop:
                post, (h, z) = self.cell.forward(embeds[i], actions[i], reset_masks[i], (h, z))
            else:
                post, (h, z) = self.cell.forward_prior(actions[i], reset_masks[i], (h, z))  # open loop: post=prior
            posts.append(post)
            states_h.append(h)
            samples.append(z)

        posts = torch.stack(posts)          # (T,BI,2S)
        states_h = torch.stack(states_h)    # (T,BI,D)
        samples = torch.stack(samples)      # (T,BI,S)
        priors = self.cell.batch_prior(states_h)  # (T,BI,2S)
        features = self.to_feature(states_h, samples)   # (T,BI,D+S)

        posts = posts.reshape(T, B, I, -1)  # (T,BI,X) => (T,B,I,X)
        states_h = states_h.reshape(T, B, I, -1)
        samples = samples.reshape(T, B, I, -1)
        priors = priors.reshape(T, B, I, -1)
        states = (states_h, samples)
        features = features.reshape(T, B, I, -1)

        return (
            priors,                      # tensor(T,B,I,2S)
            posts,                       # tensor(T,B,I,2S)
            samples,                     # tensor(T,B,I,S)
            features,                    # tensor(T,B,I,D+S)
            states,
            (h.detach(), z.detach()),
        )

    def init_state(self, batch_size):
        return self.cell.init_state(batch_size)

    def to_feature(self, h: Tensor, z: Tensor) -> Tensor:
        return torch.cat((h, z), -1)

    def feature_replace_z(self, features: Tensor, z: Tensor):
        h, _ = features.split([self.cell.deter_dim, z.shape[-1]], -1)
        return self.to_feature(h, z)

    def zdistr(self, pp: Tensor) -> D.Distribution:
        return self.cell.zdistr(pp)
    
    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        rep_loss = torch.mean(torch.clip(rep_loss, min=free))
        dyn_loss = torch.mean(torch.clip(dyn_loss, min=free))
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class RSSMCell(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm,tidy):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        self.deter_dim = deter_dim
        norm = nn.LayerNorm if layer_norm else NoNorm

        self.z_mlp = nn.Linear(stoch_dim * (stoch_discrete or 1), hidden_dim)
        self.a_mlp = nn.Linear(action_dim, hidden_dim, bias=False)  # No bias, because outputs are added
        self.in_norm = norm(hidden_dim, eps=1e-3)

        self.gru = rssm_component.GRUCellStack(hidden_dim, deter_dim, gru_layers, gru_type)

        self.prior_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.prior_norm = norm(hidden_dim, eps=1e-3)
        self.prior_mlp = nn.Linear(hidden_dim, stoch_dim * (stoch_discrete or 2))

        self.post_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.post_mlp_e = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.post_norm = norm(hidden_dim, eps=1e-3)
        self.post_mlp = nn.Linear(hidden_dim, stoch_dim * (stoch_discrete or 2))
        self.tidy=tidy

    def init_state(self, batch_size):
        device = next(self.gru.parameters()).device
        return (
            torch.zeros((batch_size, self.deter_dim), device=device),
            torch.zeros((batch_size, self.stoch_dim * (self.stoch_discrete or 1)), device=device),
        )

    def forward(self,
                embed: Tensor,                    # tensor(B,E)
                action: Tensor,                   # tensor(B,A)
                reset_mask: Tensor,               # tensor(B,1)
                in_state: Tuple[Tensor, Tensor],
                ) -> Tuple[Tensor,
                           Tuple[Tensor, Tensor]]:

        in_h, in_z = in_state
        in_h = in_h * reset_mask
        in_z = in_z * reset_mask
        B = action.shape[0]

        x = self.z_mlp(in_z) + self.a_mlp(action)  # (B,H)
        x = self.in_norm(x)
        za = F.elu(x)
        # h,[h] = self.gru(za, [in_h])                                             # (B, D)
        h=self.gru(za,in_h)
        # changed by xch
        # print(self.tidy)
        if self.tidy:
            x = self.post_mlp_e(embed)
        else:
            x = self.post_mlp_h(h) + self.post_mlp_e(embed)
        x = self.post_norm(x)
        post_in = F.elu(x)
        post = self.post_mlp(post_in)                                    # (B, S*S)
        post_distr = self.zdistr(post)
        sample = post_distr.rsample().reshape(B, -1)

        return (
            post,                         # tensor(B, 2*S)
            (h, sample),                  # tensor(B, D+S+G)
        )

    def forward_prior(self,
                      action: Tensor,                   # tensor(B,A)
                      reset_mask: Optional[Tensor],               # tensor(B,1)
                      in_state: Tuple[Tensor, Tensor],  # tensor(B,D+S)
                      ) -> Tuple[Tensor,
                                 Tuple[Tensor, Tensor]]:

        in_h, in_z = in_state
        if reset_mask is not None:
            in_h = in_h * reset_mask
            in_z = in_z * reset_mask

        B = action.shape[0]

        x = self.z_mlp(in_z) + self.a_mlp(action)  # (B,H)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h)                  # (B, D)

        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)          # (B,2S)
        prior_distr = self.zdistr(prior)
        sample = prior_distr.rsample().reshape(B, -1)

        return (
            prior,                        # (B,2S)
            (h, sample),                  # (B,D+S)
        )

    def batch_prior(self,
                    h: Tensor,     # tensor(T, B, D)
                    ) -> Tensor:
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)  # tensor(B,2S)
        return prior

    def zdistr(self, pp: Tensor) -> D.Distribution:
        # pp = post or prior
        if self.stoch_discrete:
            logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_discrete))
            distr = D.OneHotCategoricalStraightThrough(logits=logits.float())  # NOTE: .float() needed to force float32 on AMP
            distr = D.independent.Independent(distr, 1)  # This makes d.entropy() and d.kl() sum over stoch_dim
            return distr
        else:
            return diag_normal(pp)

class RSSM(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        layers_input=1,
        layers_output=1,
        rec_depth=1,
        shared=False,
        discrete=False,
        act="SiLU",
        norm="LayerNorm",
        mean_act="none",
        std_act="softplus",
        temp_post=True,
        min_std=0.1,
        cell="gru",
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._embed = embed
        # self._device = device

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        if self._shared:
            inp_dim += self._embed
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            inp_layers.append(norm(self._hidden, eps=1e-03))
            inp_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._inp_layers = nn.Sequential(*inp_layers)
        self._inp_layers.apply(tools_v3.weight_init)
