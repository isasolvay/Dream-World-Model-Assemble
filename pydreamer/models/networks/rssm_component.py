
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from torch.nn import Parameter


class GRU2Inputs(nn.Module):

    def __init__(self, input1_dim, input2_dim, mlp_dim=200, state_dim=200, num_layers=1, bidirectional=False, input_activation=F.elu):