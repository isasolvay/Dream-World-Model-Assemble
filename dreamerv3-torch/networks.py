
import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools


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
        self._device = device

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
        self._inp_layers.apply(tools.weight_init)

        if cell == "gru":
            self._cell = GRUCell(self._hidden, self._deter)
            self._cell.apply(tools.weight_init)
        elif cell == "gru_layer_norm":
            self._cell = GRUCell(self._hidden, self._deter, norm=True)
            self._cell.apply(tools.weight_init)
        else:
            raise NotImplementedError(cell)

        img_out_layers = []
        inp_dim = self._deter
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            img_out_layers.append(norm(self._hidden, eps=1e-03))
            img_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            obs_out_layers.append(norm(self._hidden, eps=1e-03))
            obs_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden