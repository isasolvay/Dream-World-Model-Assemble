
from typing import Any, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from ..tools import *
from .actorcritic import *
from .networks.common import *
from .math_functions import *
from .networks.encoders import *
from .networks.decoders import *
from .networks.rssm_component import *
from .networks.rssm import *
# from .rssm_simplified import RSSMCore, RSSMCell
from .probes import *
from . import tools_v3
from .networks import *


class WorldModel(nn.Module):

    def __init__(self, obs_space,step, conf,device):
        super().__init__()
        self._step=step
        self._conf=conf
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self._device=device
        self.deter_dim = conf.deter_dim
        self.stoch_dim = conf.stoch_dim
        self.stoch_discrete = conf.stoch_discrete
        self.kl_weight = conf.kl_weight
        self.aux_critic_weight = conf.aux_critic_weight
        
        self.wm_type=conf.wm_type
        self.kl_balance=conf.kl_balance
       
        # Encoder

        self.encoder = MultiEncoder_v2(shapes,conf)
        
        # RSSM

        self.dynamics = RSSMCore(embed_dim=self.encoder.out_dim,
                             action_dim=conf.action_dim,
                             deter_dim=conf.deter_dim,
                             stoch_dim=conf.stoch_dim,
                             stoch_discrete=conf.stoch_discrete,
                             hidden_dim=conf.hidden_dim,
                             gru_layers=conf.gru_layers,
                             gru_type=conf.gru_type,
                             layer_norm=conf.layer_norm,
                             tidy=conf.tidy)


        # Decoders for image,rewards and cont

        features_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        self.decoder = MultiDecoder(features_dim, conf)
        
        
        # Auxiliary critic

        if conf.aux_critic:
            self.ac_aux = ActorCritic(in_dim=features_dim,
                                      out_actions=conf.action_dim,
                                      layer_norm=conf.layer_norm,
                                      gamma=conf.gamma_aux,
                                      lambda_gae=conf.lambda_gae_aux,
                                      entropy_weight=conf.entropy,
                                      target_interval=conf.target_interval_aux,
                                      actor_grad=conf.actor_grad,
                                      actor_dist=conf.actor_dist,
                                      )
        else:
            self.ac_aux = None

        # Init

        for m in self.modules():
            init_weights_tf2(m)

    def init_state(self, batch_size: int) -> Tuple[Any, Any]:
        return self.dynamics.init_state(batch_size)

    def forward(self,
                obs: Dict[str, Tensor],
                in_state: Any
                ):
        loss, features, states, out_state, metrics, tensors = \
            self.training_step(obs, in_state, forward_only=True)
        return features, out_state

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      do_open_loop=False,
                      do_image_pred=False,
                      forward_only=False
                      ):
        # Encoder

        embed = self.encoder(obs)

        # RSSM

        prior, post, post_samples, features, states, out_state = \
            self.dynamics.forward(embed,
                              obs['action'],
                              obs['reset'],
                              in_state,
                              iwae_samples=iwae_samples,
                              do_open_loop=do_open_loop)

        if forward_only:
            return torch.tensor(0.0), features, states, out_state, {}, {}

        # Decoder

        loss_reconstr, metrics, tensors = self.decoder.training_step(features, obs)

        # KL loss
        d = self.dynamics.zdistr
        dprior = d(prior)
        dpost = d(post)
        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)  # (T,B,I)
        if iwae_samples == 1:
            # Analytic KL loss, standard for VAE
            if not self.kl_balance:
                loss_kl = loss_kl_exact
            else:
                loss_kl_postgrad = D.kl.kl_divergence(dpost, d(prior.detach()))
                loss_kl_priograd = D.kl.kl_divergence(d(post.detach()), dprior)
                if self.wm_type=='v2':
                    loss_kl = (1 - self.kl_balance) * loss_kl_postgrad + self.kl_balance * loss_kl_priograd
                elif self.wm_type=='v3':
                    kl_free = tools_v3.schedule(self._conf.kl_free, self._step)
                    dyn_scale = tools_v3.schedule(self._conf.dyn_scale, self._step)
                    rep_scale = tools_v3.schedule(self._conf.rep_scale, self._step)
                    # Do a clip
                    rep_loss = torch.clip(loss_kl_postgrad, min=kl_free)
                    dyn_loss = torch.clip(loss_kl_priograd, min=kl_free)
                    loss_kl = dyn_scale * dyn_loss + rep_scale * rep_loss
        else:
            # Sampled KL loss, for IWAE
            z = post_samples.reshape(dpost.batch_shape + dpost.event_shape)
            loss_kl = dpost.log_prob(z) - dprior.log_prob(z)

        # Auxiliary critic loss

        if self.ac_aux:
            features_tb = features.select(2, 0)  # (T,B,I) => (T,B) - assume I=1
            (_, loss_critic_aux), metrics_ac, tensors_ac = \
                self.ac_aux.training_step(features_tb,
                                          obs['action'][1:],
                                          obs['reward'],
                                          obs['terminal'])
            metrics.update(loss_critic_aux=metrics_ac['loss_critic'],
                           policy_value_aux=metrics_ac['policy_value_im'])
            tensors.update(policy_value_aux=tensors_ac['value'])
        else:
            loss_critic_aux = 0.0

        # Total loss

        assert loss_kl.shape == loss_reconstr.shape
        loss_model_tbi = self.kl_weight * loss_kl + loss_reconstr
        loss_model = -logavgexp(-loss_model_tbi, dim=2).mean()
        # loss = loss_model.mean() + self.aux_critic_weight * loss_critic_aux
