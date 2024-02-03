
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
        
        if self.wm_type=='v2':
            loss_model, features, states, out_state, metrics, tensors = \
                self.wm.training_step(obs,
                                    in_state,
                                    iwae_samples=iwae_samples,
                                    do_open_loop=do_open_loop,
                                    do_image_pred=do_image_pred)
        elif self.wm_type=='v3':
            # loss_model,features, post, context, metrics,tensors=self.wm.training_step(obs,do_image_pred=do_image_pred)
            # out_state={key: tensor[-1] for key, tensor in post.items()}
            loss_model, features, states, out_state, metrics, tensors = \
                self.wm.training_step(obs,
                                    in_state,
                                    iwae_samples=iwae_samples,
                                    do_open_loop=do_open_loop,
                                    do_image_pred=do_image_pred)

        # Map probe

        features_probe = features.detach() if not self.probe_gradients else features
        loss_probe, metrics_probe, tensors_probe = self.probe_model.training_step(features_probe, obs)
        metrics.update(**metrics_probe)
        tensors.update(**tensors_probe)

        # Policy
        
        in_state_dream: StateB = map_structure(states, lambda x: flatten_batch(x.detach())[0]) 
        # type: ignore  # (T,B,I) => (TBI)
        # if self.wm_type=='v2':
        features_dream, actions_dream, rewards_dream, terminals_dream, states_dream = \
        self.dream(in_state_dream, H, self.ac._conf.actor_grad == 'dynamics')  # (H+1,TBI,D)
        
        # elif self.wm_type=='v3':
        # #     # states=post
        # #     flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        # #     in_state_dream = {k: flatten(v) for k, v in states.items()}
        # #     # in_state_dream={key: tensor[0] for key, tensor in states.items()}
        # #     # in_state_dream=map_structure(states, lambda x: flatten_batch(x.detach())[0])
        # # # Note features_dream includes the starting "real" features at features_dream[0]
        #     features_dream, actions_dream, rewards_dream, terminals_dream, states_dream = \
        #     self.dream(in_state_dream, H, self.ac._conf.actor_grad == 'dynamics')
            # states_dream={key: tensor.detach() for key, tensor in states_dream.items()}
        if self.wm_type=="v2":
            (loss_actor, loss_critic), metrics_ac, tensors_ac = \
                self.ac.training_step(features_dream.detach(),
                                    actions_dream.detach(),
                                    rewards_dream.mean.detach(),
                                    terminals_dream.mean.detach())
            # tensors.update(policy_value=unflatten_batch(tensors_ac['value'][0], (T, B, I)).mean(-1))
        elif self.wm_type=="v3":
            #  (loss_actor, loss_critic), metrics_ac, tensors_ac = \
            #     self.ac.training_step(features_dream.detach(),
            #                         actions_dream.detach(),
            #                         rewards_dream.mean.detach(),
            #                         terminals_dream.mean.detach(),
            #                         # states_dream
            #                         )
            (loss_actor, loss_critic), metrics_ac, tensors_ac = \
                self.ac.training_step(features_dream.detach(),
                                    actions_dream.detach(),
                                    rewards_dream.mode().detach(),
                                    terminals_dream.mode().detach())
            # tensors.update(policy_value=unflatten_batch(tensors_ac['value'][0], (T, B, I)).mean(-1))
        metrics.update(**metrics_ac)
       

        # Dream for a log sample.
        ## 扰动和刻意的action应该都可以在这里加。
        dream_tensors = {}
        if self.wm=="v2":
            if do_dream_tensors and self.wm.decoder.image is not None:
                with torch.no_grad():  # careful not to invoke modules first time under no_grad (https://github.com/pytorch/pytorch/issues/60164)
                    # The reason we don't just take real features_dream is because it's really big (H*T*B*I),
                    # and here for inspection purposes we only dream from first step, so it's (H*B).
                    # Oh, and we set here H=T-1, so we get (T,B), and the dreamed experience aligns with actual.
                    # 这里实际做的时候，T=1，只从第一步想象
                    in_state_dream: StateB = map_structure(states, lambda x: x.detach()[0, :, 0])  # type: ignore  # (T,B,I) => (B)
                    ## 基本上只改了这一步
                    # non_zero_indices = torch.nonzero(in_state_dream[1])

                    # print(non_zero_indices)
                    features_dream, actions_dream, rewards_dream, terminals_dream,states_dream = self.dream(in_state_dream, T - 1)  # H = T-1
                    # features_dream, actions_dream, rewards_dream, terminals_dream = self.dream_cond_action(in_state_dream, obs['action'])
                    image_dream = self.wm.decoder.image.forward(features_dream)
                    ## 拿Dreamer_agent的数据只训练actor_critic
                    # _, _, tensors_ac = self.ac.training_step(features_dream, actions_dream[1:,:,:], rewards_dream.mean, terminals_dream.mean, log_only=True)
                    _, _, tensors_ac = self.ac.training_step(features_dream, actions_dream, rewards_dream.mean, terminals_dream.mean, log_only=True)
                    # The tensors are intentionally named same as in tensors, so the logged npz looks the same for dreamed or not
                    # dream_tensors = dict(action_pred=actions_dream,  # first action is real from previous step
                    #                      reward_pred=rewards_dream.mean,
                    #                      terminal_pred=terminals_dream.mean,
                    #                      image_pred=image_dream,
                    #                      **tensors_ac)
                    dream_tensors = dict(action_pred=torch.cat([obs['action'][:1], actions_dream]),  # first action is real from previous step
                                        reward_pred=rewards_dream.mean,
                                        terminal_pred=terminals_dream.mean,
                                        image_pred=image_dream,
                                        **tensors_ac)
                    assert dream_tensors['action_pred'].shape == obs['action'].shape
                    assert dream_tensors['image_pred'].shape == obs['image'].shape
        elif self.wm_type=="v3":
            if do_dream_tensors:
                with torch.no_grad():  # careful not to invoke modules first time under no_grad (https://github.com/pytorch/pytorch/issues/60164)
                    # The reason we don't just take real features_dream is because it's really big (H*T*B*I),
                    # and here for inspection purposes we only dream from first step, so it's (H*B).
                    # Oh, and we set here H=T-1, so we get (T,B), and the dreamed experience aligns with actual.
                    # 这里实际做的时候，T=1，只从第一步想象
                    # in_state_dream={key: tensor[0] for key, tensor in states.items()}
                    in_state_dream: StateB = map_structure(states, lambda x: x.detach()[0, :, 0])
                    ## 基本上只改了这一步
                    # non_zero_indices = torch.nonzero(in_state_dream[1])

                    # print(non_zero_indices)
                    features_dream, actions_dream, rewards_dream, terminals_dream,states_dream = self.dream(in_state_dream, T - 1)  # H = T-1
                    # features_dream, actions_dream, rewards_dream, terminals_dream = self.dream_cond_action(in_state_dream, obs['action'])
                    image_dream = self.wm.decoder.image.forward(features_dream)
                    # image_dream = self.wm.decoder.image.forward(features_dream)
                    _, _, tensors_ac = self.ac.training_step(features_dream, actions_dream, rewards_dream.mode(), terminals_dream.mode(), log_only=True)
                    ## 拿Dreamer_agent的数据只训练actor_critic
                    # _, _, tensors_ac = self.ac.training_step(features_dream, actions_dream[1:,:,:], rewards_dream.mean, terminals_dream.mean, log_only=True)
                    # _, _, tensors_ac = self.ac.training_step(features_dream, actions_dream, rewards_dream.mean, terminals_dream.mean, log_only=True)
                    # (actor_loss,value_loss),features, states, actions, metrics=self.ac.training_step(features_dream.detach(),
                    #                 actions_dream.detach(),
                    #                 rewards_dream.detach(),
                    #                 terminals_dream.detach(),
                    #                 states_dream)
                    # The tensors are intentionally named same as in tensors, so the logged npz looks the same for dreamed or not
                    # dream_tensors = dict(action_pred=actions_dream,  # first action is real from previous step
                    #                      reward_pred=rewards_dream.mean,
                    #                      terminal_pred=terminals_dream.mean,
                    #                      image_pred=image_dream,
                    #                      **tensors_ac)
                    dream_tensors = dict(action_pred=torch.cat([obs['action'][:1], actions_dream]),  # first action is real from previous step
                                        reward_pred=rewards_dream.mode(),
                                        terminal_pred=terminals_dream.mode(),
                                        image_pred=image_dream,
                                        )
                    assert dream_tensors['action_pred'].shape == obs['action'].shape
                    assert dream_tensors['image_pred'].shape == obs['image'].shape
        if not self.probe_gradients:
            losses = (loss_model, loss_probe, loss_actor, loss_critic)
        else:
            losses = (loss_model + loss_probe, loss_actor, loss_critic)
        return losses, out_state, metrics, tensors, dream_tensors