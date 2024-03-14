
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

        # Metrics

        with torch.no_grad():
            # loss_kl = -logavgexp(-loss_kl_exact, dim=2)  # Log exact KL loss even when using IWAE, it avoids random negative values
            loss_kl = -logavgexp(-loss_kl, dim=2)
            entropy_prior = dprior.entropy().mean(dim=2)
            entropy_post = dpost.entropy().mean(dim=2)
            tensors.update(loss_kl=loss_kl.detach(),
                           entropy_prior=entropy_prior,
                           entropy_post=entropy_post)
            metrics.update(loss_model=loss_model.mean(),
                           loss_kl=loss_kl.mean(),
                           entropy_prior=entropy_prior.mean(),
                           entropy_post=entropy_post.mean())
            if self.wm_type=='v3':
                metrics["kl_free"] = kl_free
                metrics["dyn_scale"] =dyn_scale
                metrics["rep_scale"] = rep_scale
                metrics["loss_dyn"] = to_np(torch.mean(dyn_loss))
                metrics["loss_rep"] = to_np(torch.mean(rep_loss))

        # Predictions

        if do_image_pred:
            with torch.no_grad():
                prior_samples = self.dynamics.zdistr(prior).sample().reshape(post_samples.shape)
                features_prior = self.dynamics.feature_replace_z(features, prior_samples)
                # Decode from prior(就是没有看到xt，凭借ht直接给出的预测)
                _, mets, tens = self.decoder.training_step(features_prior, obs, extra_metrics=True)
                metrics_logprob = {k.replace('loss_', 'logprob_'): v for k, v in mets.items() if k.startswith('loss_')}
                tensors_logprob = {k.replace('loss_', 'logprob_'): v for k, v in tens.items() if k.startswith('loss_')}
                tensors_pred = {k.replace('_rec', '_pred'): v for k, v in tens.items() if k.endswith('_rec')}
                metrics.update(**metrics_logprob)   # logprob_image, ...
                tensors.update(**tensors_logprob)  # logprob_image, ...
                tensors.update(**tensors_pred)  # image_pred, ...

        return loss_model, features, states, out_state, metrics, tensors
    
    
class WorldModel_v3(nn.Module):
    def __init__(self, obs_space, step, conf,device):
        # super(WorldModel_v3, self).__init__()
        super().__init__()
        self._step = step
        # self._use_amp = True if conf.precision == 16 else False
        self._conf = conf
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self._device=device
        
        self.deter_dim = conf.deter_dim
        self.stoch_dim = conf.stoch_dim
        self.stoch_discrete = conf.stoch_discrete
        self.kl_weight = conf.kl_weight
        self.aux_critic_weight = conf.aux_critic_weight
        self.kl_balance=conf.kl_balance
        
        # Encoder
        # self.encoder = MultiEncoder_v3(shapes, conf)
        self.encoder = MultiEncoder_v2(shapes,conf)
        # RSSM
        # self.embed_size = self.encoder.out_dim
        # # self.dynamics = RSSM(
        # #     conf.dyn_stoch,
        # #     conf.dyn_deter,
        # #     conf.dyn_hidden,
        # #     conf.dyn_input_layers,
        # #     conf.dyn_output_layers,
        # #     conf.dyn_rec_depth,
        # #     conf.dyn_shared,
        # #     conf.dyn_discrete,
        # #     conf.act,
        # #     conf.norm,
        # #     conf.dyn_mean_act,
        # #     conf.dyn_std_act,
        # #     conf.dyn_temp_post,
        # #     conf.dyn_min_std,
        # #     conf.dyn_cell,
        # #     conf.unimix_ratio,
        # #     conf.initial,
        # #     #为啥原文件里没有这个
        # #     # conf.num_actions,
        # #     conf.action_dim,
        # #     self.embed_size,
        # #     conf.device,
        # # )
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
        # dECODERS FOR IMAGE,REWARDS and counts
        if conf.dyn_discrete:
            # features_dim = conf.dyn_stoch * conf.dyn_discrete + conf.dyn_deter
            features_dim=conf.deter_dim+conf.stoch_dim * (conf.stoch_discrete or 1)
        else:
            features_dim = conf.dyn_stoch + conf.dyn_deter
        self.decoder = MultiDecoder_v2(features_dim, conf)
        # self.heads = nn.ModuleDict()
        
        #     features_dim=conf.deter_dim+conf.stoch_dim
        # self.heads["decoder"] = MultiDecoder_v3(
        #     features_dim, shapes, **conf.decoder
        # )
        # if conf.reward_head == "symlog_disc":
        #     self.heads["reward"] = MLP_v3(
        #         features_dim,  # pytorch version
        #         (255,),
        #         conf.reward_layers,
        #         conf.units,
        #         conf.act,
        #         conf.norm,
        #         dist=conf.reward_head,
        #         outscale=0.0,
        #         device=self._device,
        #     )
        # else:
        #     self.heads["reward"] = MLP_v3(
        #         features_dim,  # pytorch version
        #         [],
        #         conf.reward_layers,
        #         conf.units,
        #         conf.act,
        #         conf.norm,
        #         dist=conf.reward_head,
        #         outscale=0.0,
        #         device=self._device,
        #     )
        # self.heads["terminal"] = MLP_v3(
        #     features_dim,  # pytorch version
        #     [],
        #     conf.terminal_layers,
        #     conf.units,
        #     conf.act,
        #     conf.norm,
        #     dist="binary",
        #     device=self._device,
        # )
        # for name in conf.grad_heads:
        #     assert name in self.heads, name
        # # self._model_opt = tools_v3.Optimizer(
        # #     "model",
        # #     self.parameters(),
        # #     conf.model_lr,
        # #     conf.opt_eps,
        # #     conf.grad_clip,
        # #     conf.weight_decay,
        # #     opt=conf.opt,
        # #     use_amp=self._use_amp,
        # # )