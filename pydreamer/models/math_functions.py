
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor, Size

from .networks import rssm_component

def flatten(x: Tensor) -> Tensor:
    # (T, B, ...) => (T*B, ...)
    return torch.reshape(x, (-1,) + x.shape[2:])


def unflatten(x: Tensor, n: int) -> Tensor:
    # (T*B, ...) => (T, B, ...)
    return torch.reshape(x, (n, -1) + x.shape[1:])


def flatten_batch(x: Tensor, nonbatch_dims=1) -> Tuple[Tensor, Size]:
    # (b1,b2,..., X) => (B, X)
    if nonbatch_dims > 0:
        batch_dim = x.shape[:-nonbatch_dims]
        x = torch.reshape(x, (-1,) + x.shape[-nonbatch_dims:])
        return x, batch_dim
    else:
        batch_dim = x.shape
        x = torch.reshape(x, (-1,))
        return x, batch_dim


def unflatten_batch(x: Tensor, batch_dim: Union[Size, Tuple]) -> Tensor:
    # (B, X) => (b1,b2,..., X)
    x = torch.reshape(x, batch_dim + x.shape[1:])
    return x


def insert_dim(x: Tensor, dim: int, size: int) -> Tensor:
    """Inserts dimension and expands it to size."""
    x = x.unsqueeze(dim)
    x = x.expand(*x.shape[:dim], size, *x.shape[dim + 1:])
    return x


def diag_normal(x: Tensor, min_std=0.1, max_std=2.0):
    # DreamerV2:
    # std = {
    #     'softplus': lambda: tf.nn.softplus(std),
    #     'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
    # }[self.std_act]()
    # std = std + self.min_std
    mean, std = x.chunk(2, -1)
    # std = F.softplus(std) + min_std
    std = max_std * torch.sigmoid(std) + min_std
    return D.independent.Independent(D.normal.Normal(mean, std), 1)


def normal_tanh(x: Tensor, min_std=0.01, max_std=1.0):
    # Normal(tanh(x))
    mean_, std_ = x.chunk(2, -1)
    mean = torch.tanh(mean_)
    std = max_std * torch.sigmoid(std_) + min_std
    normal = D.normal.Normal(mean, std)
    normal = D.independent.Independent(normal, 1)
    return normal


def tanh_normal(x: Tensor):
    # TanhTransform(Normal(5 tanh(x/5)))
    mean_, std_ = x.chunk(2, -1)
    mean = 5 * torch.tanh(mean_ / 5)  # clip tanh arg to (-5, 5)
    std = F.softplus(std_) + 0.1  # min_std = 0.1
    normal = D.normal.Normal(mean, std)
    normal = D.independent.Independent(normal, 1)
    tanh = D.TransformedDistribution(normal, [D.TanhTransform()])
    tanh.entropy = normal.entropy  # HACK: need to implement correct tanh.entorpy (need Jacobian of TanhTransform?)
    return tanh


def init_weights_tf2(m):
    # Match TF2 initializations
    if type(m) in {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)