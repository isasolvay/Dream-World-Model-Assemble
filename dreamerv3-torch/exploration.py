import torch
from torch import nn
from torch import distributions as torchd

import models
import networks
import tools


class Random(nn.Module):
    def __init__(self, config, act_space):
        super(Random, self).__init__()
        self._config = config
        self._act_space = act_space

    def actor(self, feat):
        if self._config.actor_dist == "onehot":
            return tools.OneHotDist(
                torch.zeros(self._config.num_actions)
                .repeat(self._config.envs, 1)
                .to(self._config.device)
            )
        else:
            return torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(self._act_space.low)
                    .repeat(self._config.envs, 1)
                    .to(self._config.device),
                    torch.Tensor(self._act_space.high)
                    .repeat(self._config.envs, 1)
                    .to(self._config.device),
                ),
                1,
            )

    def train(self, start, context, data):
        return None, {}


class Plan2Explore(nn.Module):
    def __init__(self, config, world_model, reward=None):
        super(Plan2Explore, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        self._reward = reward
        self._behavior = models.ImagBehavior(config, world_model)
        self.actor = self._behavior.actor
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            stoch = config.dyn_stoch * config.dyn_discrete
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            stoch = config.dyn_stoch
        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.dyn_deter,
            "feat": config.dyn_stoch + config.dyn_deter,
        }[self._config.disag_target]
        kw = dict(
            inp_dim=feat_size + config.num_actions
            if 