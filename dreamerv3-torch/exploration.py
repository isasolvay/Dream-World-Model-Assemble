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
  