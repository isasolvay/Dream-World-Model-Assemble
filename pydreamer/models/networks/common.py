
import math
import numpy as np
import re
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch import distributions as torchd

from ..tools_v3 import *
from .. math_functions import *

# This is a work-in-progress attempt to use type aliases to indicate the shapes of tensors.
# T = 50         (TBTT length)
# B = 50         (batch size)
# I = 1/3/10     (IWAE samples)
# A = 3          (action dim)
# E              (embedding dim)
# F = 2048+32    (feature_dim)
# H = 10         (dream horizon)
# J = H+1 = 11
# M = T*B*I = 2500
TensorTBCHW = Tensor
TensorTB = Tensor
TensorTBE = Tensor
TensorTBICHW = Tensor
TensorTBIF = Tensor
TensorTBI = Tensor
TensorJMF = Tensor
TensorJM2 = Tensor
TensorHMA = Tensor
TensorHM = Tensor
TensorJM = Tensor

IntTensorTBHW = Tensor
StateB = Tuple[Tensor, Tensor]
StateTB = Tuple[Tensor, Tensor]


class MLP_v2(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, hidden_layers, layer_norm, activation=nn.ELU):
        super().__init__()
        self.out_dim = out_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = []
        dim = in_dim
        for i in range(hidden_layers):
            layers += [
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()
            ]
            dim = hidden_dim
        layers += [
            nn.Linear(dim, out_dim),
        ]
        if out_dim == 1:
            layers += [
                nn.Flatten(0),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y
    
class MLP_v3(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm="LayerNorm",
        dist="normal",
        std=1.0,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
    ):
        super(MLP_v3, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        self._layers = layers
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._dist = dist
        self._std = std
        self._symlog_inputs = symlog_inputs
        self._device = device

        layers = []
        for index in range(self._layers):
            layers.append(nn.Linear(inp_dim, units, bias=False))
            layers.append(norm(units, eps=1e-03))
            layers.append(act())
            if index == 0:
                inp_dim = units
        self.layers = nn.Sequential(*layers)
        self.layers.apply(weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = symlog(x)
        out = self.layers(x)
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)
        
    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        return -output.log_prob(target)
        
    def training_step(self, features,target: Tensor):
    # if self._dist=='binary':
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it
        if len(target.shape)==3:
            target= target.unsqueeze(-1)

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        decoded = decoded.mode().mean(dim=-2)

        assert len(loss_tbi.shape) == 3
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == 2
        return loss_tbi, loss_tb, decoded

    def dist(self, dist, mean, std, shape):
        if dist == "normal":
            return ContDist(
                torchd.independent.Independent(
                    torchd.normal.Normal(mean, std), len(shape)
                )
            )
        if dist == "huber":
            return ContDist(
                torchd.independent.Independent(
                    UnnormalizedHuber(mean, std, 1.0), len(shape)
                )