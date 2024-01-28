
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
from .tools_v3 import *
from .networks import *
from .world_models import *


class Dreamer_agent(nn.Module):

    def __init__(self, conf,obs_space,act_space,step,device=None):
        super().__init__()
        assert conf.action_dim > 0, "Need to set action_dim to match environment"
        features_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        self.iwae_samples = conf.iwae_samples
        self.imag_horizon = conf.imag_horizon
        if device:
            self._device=device
        else:
            self._device=conf.device

        # World model
        self.wm_type=conf.wm_type
        # if self.wm_type=='v2':
        self.wm = WorldModel(obs_space,step,conf,self._device)
        # elif self.wm_type=='v3':
            # self.wm = WorldModel_v3(obs_space,step,conf,self._device)

        # Actor critic
        
        # if self.wm_type=='v2':
        self.ac = ActorCritic(conf,self.wm,self._device)
        # elif self.wm_type=='v3':
        #     self.ac=ActorCritic_v3(conf,self.wm,self._device)
            #  self.ac = ActorCritic_v2(conf
            #                   )


        
        # Map probe

        # if conf.probe_model == 'map':
        #     probe_model = MapProbeHead(features_dim + 4, conf)
        # elif conf.probe_model == 'goals':
        #     probe_model = GoalsProbe(features_dim, conf)
        # elif conf.probe_model == 'map+goals':
        #     probe_model = MapGoalsProbe(features_dim, conf)
        # elif conf.probe_model == 'none':
        probe_model = NoProbeHead()
        # else:
        #     raise NotImplementedError(f'Unknown probe_model={conf.probe_model}')
        self.probe_model = probe_model
        self.probe_gradients = conf.probe_gradients

    def init_optimizers(self, lr, lr_actor=None, lr_critic=None, eps=1e-5,ac_eps=1e-5):
        if not self.probe_gradients:
            optimizer_wm = torch.optim.AdamW(self.wm.parameters(), lr=lr, eps=eps)
            optimizer_probe = torch.optim.AdamW(self.probe_model.parameters(), lr=lr, eps=eps)
            optimizer_actor = torch.optim.AdamW(self.ac.actor.parameters(), lr=lr_actor or lr, eps=ac_eps)
            optimizer_critic = torch.optim.AdamW(self.ac.critic.parameters(), lr=lr_critic or lr, eps=ac_eps)
            return optimizer_wm, optimizer_probe, optimizer_actor, optimizer_critic
        else:
            optimizer_wmprobe = torch.optim.AdamW(self.wm.parameters(), lr=lr, eps=eps)
            optimizer_actor = torch.optim.AdamW(self.ac.actor.parameters(), lr=lr_actor or lr, eps=ac_eps)
            optimizer_critic = torch.optim.AdamW(self.ac.critic.parameters(), lr=lr_critic or lr, eps=ac_eps)
            return optimizer_wmprobe, optimizer_actor, optimizer_critic

    def grad_clip(self, grad_clip, grad_clip_ac=None):
        if not self.probe_gradients:
            grad_metrics = {
                'grad_norm': nn.utils.clip_grad_norm_(self.wm.parameters(), grad_clip),
                'grad_norm_probe': nn.utils.clip_grad_norm_(self.probe_model.parameters(), grad_clip),
                'grad_norm_actor': nn.utils.clip_grad_norm_(self.ac.actor.parameters(), grad_clip_ac or grad_clip),
                'grad_norm_critic': nn.utils.clip_grad_norm_(self.ac.critic.parameters(), grad_clip_ac or grad_clip),
            }
        else:
            grad_metrics = {
                'grad_norm': nn.utils.clip_grad_norm_(self.wm.parameters(), grad_clip),
                'grad_norm_actor': nn.utils.clip_grad_norm_(self.ac.actor.parameters(), grad_clip_ac or grad_clip),
                'grad_norm_critic': nn.utils.clip_grad_norm_(self.ac.critic.parameters(), grad_clip_ac or grad_clip),
            }
        return grad_metrics

    def init_state(self, batch_size: int):
        return self.wm.init_state(batch_size)

    def inference(self,
                  obs: Dict[str, Tensor],
                  in_state: Any,
                  ) -> Tuple[D.Distribution, Any, Dict]:
        assert 'action' in obs, 'Observation should contain previous action'
        act_shape = obs['action'].shape
        assert len(act_shape) == 3 and act_shape[0] == 1, f'Expected shape (1,B,A), got {act_shape}'

        # Forward (world model)

        features, out_state = self.wm.forward(obs, in_state)

        # Forward (actor critic)
        if self.wm_type=='v2':
            
            feature = features[:, :, 0]  # (T=1,B,I=1,F) => (1,B,F) ## features.shape(1,1,1,2048)T*B*I*(D+S)
            action_distr=self.ac.forward_actor(feature)
            value=self.ac.forward_value(feature)
            metrics = dict(policy_value=value.detach().mean())
        elif self.wm_type=='v3':
            feature = features[:, :, 0]
            # feature = features[0, :, :] ##features.shape(1,1,1536)T*B*(D+S)
            action_distr = self.ac.forward_actor(feature)  # (1,B,A)
            value = self.ac.critic(feature)  # (1,B)
            metrics={}
            metrics.update(tensorstats(value.mode().detach(), "policy_value"))
            # metrics = dict(policy_value=value.detach().mean())
            # feature = features[:, :, 0]  # (T=1,B,I=1,F) => (1,B,F) ## features.shape(1,1,1,2048)T*B*I*(D+S)
            # action_distr=self.ac.forward_actor(feature)
            # value=self.ac.forward_value(feature)
            # metrics = dict(policy_value=value.detach().mean())
        
        return action_distr, out_state, metrics

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: Optional[int] = None,
                      imag_horizon: Optional[int] = None,
                      do_open_loop=False,
                      do_image_pred=False,
                      do_dream_tensors=False,
                      ):
        assert 'action' in obs, '`action` required in observation'
        assert 'reward' in obs, '`reward` required in observation'
        assert 'reset' in obs, '`reset` required in observation'
        assert 'terminal' in obs, '`terminal` required in observation'
        iwae_samples = int(iwae_samples or self.iwae_samples)
        imag_horizon = int(imag_horizon or self.imag_horizon)
        T, B = obs['action'].shape[:2]
        I, H = iwae_samples, imag_horizon
        # print(T,B,I,H)
        # World model
        