
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import copy
from torch import Tensor

from .math_functions import *
from .networks.common import *
from.tools_v3 import *

class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()

class ActorCritic(nn.Module):

    def __init__(self,
                 conf,world_model,device
                 ):
        super().__init__()
        ## feature_dim (h,z)
        self._conf=conf
        self.wm_type=conf.wm_type
        self._world_model=world_model
        self._device=device
        # self.action_dim = conf.action_dim
        # self.discount = conf.discount
        # self.lambda_ = conf.lambda_gae
        # self.entropy_weight = conf.actor_entropy
        # self.slow_value_target=conf.slow_value_target
        # self.slow_target_update=conf.slow_target_update
        # self.slow_target_fraction=conf.slow_target_fraction
        # self.actor_grad = conf.actor_grad
        # self.actor_dist = conf.actor_dist
        actor_out_dim = conf.action_dim if conf.actor_dist in ["normal_1", "onehot", "onehot_gumbel"]  else 2 * conf.action_dim
        feat_size = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        self.feat_size = feat_size
        hidden_layers=4
        
        
        if self.wm_type=='v2':
            self.actor = MLP_v2(feat_size, actor_out_dim, conf.hidden_dim, hidden_layers, conf.layer_norm)
            self.critic = MLP_v2(feat_size, 1,  conf.hidden_dim, hidden_layers, conf.layer_norm)
        elif self.wm_type=='v3':
            self.actor = ActionHead(
            feat_size,
            actor_out_dim,
            conf.actor_layers,
            conf.units,
            conf.act,
            conf.norm,
            conf.actor_dist,
            conf.actor_init_std,
            conf.actor_min_std,
            conf.actor_max_std,
            conf.actor_temp,
            outscale=1.0,
            unimix_ratio=conf.action_unimix_ratio,
            )   
            if self._conf.reward_EMA:
                self.reward_ema = RewardEMA(device=self._device)
            if conf.value_head == "symlog_disc":
                self.critic = MLP_v3(
                    feat_size,
                    (255,),
                    conf.value_layers,
                    conf.units,
                    conf.act,
                    conf.norm,
                    conf.value_head,
                    outscale=0.0,
                    device=self._device,
                )
            else:
                self.critic = MLP_v3(
                    feat_size,
                    [],
                    conf.value_layers,
                    conf.units,
                    conf.act,
                    conf.norm,
                    conf.value_head,
                    outscale=0.0,
                    device=self._device,
                )
        # self.critic_target = MLP_v2(feat_size, 1,  conf.hidden_dim, conf.hidden_layers, conf.layer_norm)
        # ## Here is a change! Orginally false, but I change it to true
        # # self.critic_target.requires_grad_(False)
        # self.critic_target.requires_grad_(True)
        if conf.slow_value_target:
            self._slow_value = copy.deepcopy(self.critic)
            self._updates = 0

    def forward_actor(self, features: Tensor) -> D.Distribution:
        if self.wm_type=='v2':
            y = self.actor.forward(features).float()  # .float() to force float32 on AMP
            
            if self._conf.actor_dist == 'onehot':
                return D.OneHotCategorical(logits=y)
            
            if self._conf.actor_dist == 'normal_tanh':
                return normal_tanh(y)

            if self._conf.actor_dist == 'tanh_normal':
                return tanh_normal(y)
        elif self.wm_type=='v3':
            y = self.actor.forward(features)
            return y

        assert False, self._conf.actor_dist

    def forward_value(self, features: Tensor) -> Tensor:
        y = self.critic.forward(features)
        return y

    def training_step(self,
                      features: TensorJMF,
                      actions: TensorHMA,
                      rewards: TensorJM,
                      terminals: TensorJM,
                      log_only=False
                      ):
        """
        The ordering is as follows:
            features[0] 
            -> actions[0] -> rewards[1], terminals[1], features[1]
            -> actions[1] -> ...
            ...
            -> actions[H-1] -> rewards[H], terminals[H], features[H]
        """
        if not log_only:
            # 每轮都更新一点点
            # if self._updates % self.target_interval == 0:
                # self.update_critic_target()
            self._update_slow_target()
        self._updates += 1
        
        # reward1: TensorHM = rewards[1:]
        # terminal0: TensorHM = terminals[:-1]
        # terminal1: TensorHM = terminals[1:]
        # if self._conf.wm_type=='v3':
        #     reward1=reward1.squeeze(-1)
        #     terminal0=terminal0.squeeze(-1)
        #     terminal1=terminal1.squeeze(-1)
        
        # # GAE from https://arxiv.org/abs/1506.02438 eq (16)
        # #   advantage_gae[t] = advantage[t] + (discount lambda) advantage[t+1] + (discount lambda)^2 advantage[t+2] + ...

        # value_t: TensorJM = self._slow_value.forward(features)
        # value0t: TensorHM = value_t[:-1]
        # value1t: TensorHM = value_t[1:]
        # # TD error=r+\discount*V(s')-V(s)
        # advantage = - value0t + reward1 + self._conf.discount * (1.0 - terminal1) * value1t
        # advantage_gae = []
        # agae = None
        # # GAE的累加
        # for adv, term in zip(reversed(advantage.unbind()), reversed(terminal1.unbind())):
        #     if agae is None:
        #         agae = adv
        #     else:
        #         agae = adv + self._conf.lambda_gae * self._conf.discount * (1.0 - term) * agae
        #     advantage_gae.append(agae)
        # advantage_gae.reverse()
        # advantage_gae = torch.stack(advantage_gae)
        # # Note: if lambda=0, then advantage_gae=advantage, then value_target = advantage + value0t = reward + discount * value1t
        # value_target = advantage_gae + value0t

        # # When calculating losses, should ignore terminal states, or anything after, so:
        # #   reality_weight[i] = (1-terminal[0]) (1-terminal[1]) ... (1-terminal[i])
        # # Note this takes care of the case when initial state features[0] is terminal - it will get weighted by (1-terminals[0]).
        # reality_weight = (1 - terminal0).log().cumsum(dim=0).exp()
        
        actor_ent = self.forward_actor(features[:-1]).entropy()
        # state_ent = self._world_model.dynamics.get_dist(states).entropy()
        state_ent=0
        value_target, reality_weight, base = self._compute_target(
            features, actions, rewards, terminals,actor_ent, state_ent
        )

        # Critic loss
        
        loss_critic,critic_mets,tensors=self._compute_critic_loss(
        features,
        actions,
        value_target,
        reality_weight)

        # value: TensorJM = self.critic.forward(features)
        # value0: TensorHM = value[:-1]
        # loss_critic = 0.5 * torch.square(value_target.detach() - value0)
        # loss_critic = (loss_critic * reality_weight).mean()

        # Actor loss
        
        #actor_loss