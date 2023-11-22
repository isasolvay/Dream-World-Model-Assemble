
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