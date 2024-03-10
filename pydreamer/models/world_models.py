
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
from . import tools_v3
from .networks import *


class WorldModel(nn.Module):

    def __init__(self, obs_space,step, conf,device):
        super().__init__()
        self._step=step
        self._conf=conf
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self._device=device
        self.deter_dim = conf.deter_dim
        self.stoch_dim = conf.stoch_dim
        self.stoch_discrete = conf.stoch_discrete
        self.kl_weight = conf.kl_weight
        self.aux_critic_weight = conf.aux_critic_weight
        
        self.wm_type=conf.wm_type