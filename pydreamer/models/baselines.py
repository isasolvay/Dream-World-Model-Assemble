
from typing import Any, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from ..tools import *
from .actorcritic import *
from .networks.common import *
from .math_functions import *
from .networks.decoders import *
from .networks.rssm_component import *
from .networks.rssm import *


class WorldModelProbe(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.probe_gradients = conf.probe_gradients

        # World model

        if conf.model == 'vae':
            self.wm = VAEWorldModel(conf)
        elif conf.model == 'gru_vae':
            self.wm = GRUVAEWorldModel(conf)
        elif conf.model == 'transformer_vae':
            self.wm = TransformerVAEWorldModel(conf)
        elif conf.model == 'gru_probe':
            self.wm = GRUEncoderOnly(conf)
        else:
            raise ValueError(conf.model)

        # Map probe

        if conf.probe_model == 'map':
            probe_model = MapProbeHead(self.wm.out_dim + 4, conf)
        elif conf.probe_model == 'goals':
            probe_model = GoalsProbe(self.wm.out_dim, conf)
        elif conf.probe_model == 'map+goals':
            probe_model = MapGoalsProbe(self.wm.out_dim, conf)
        elif conf.probe_model == 'none':
            probe_model = NoProbeHead()
        else:
            raise NotImplementedError(f'Unknown probe_model={conf.probe_model}')
        self.probe_model = probe_model

        # Use TF2 weight initialization (TODO: Dreamer is missing this on probe model)

        for m in self.modules():
            init_weights_tf2(m)

    def init_optimizers(self, lr, lr_actor=None, lr_critic=None, eps=1e-5):
        if not self.probe_gradients:
            optimizer_wm = torch.optim.AdamW(self.wm.parameters(), lr=lr, eps=eps)
            optimizer_probe = torch.optim.AdamW(self.probe_model.parameters(), lr=lr, eps=eps)
            return optimizer_wm, optimizer_probe
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, eps=eps)
            return (optimizer,)

    def grad_clip(self, grad_clip, grad_clip_ac=None):
        if not self.probe_gradients:
            grad_metrics = {
                'grad_norm': nn.utils.clip_grad_norm_(self.wm.parameters(), grad_clip),
                'grad_norm_probe': nn.utils.clip_grad_norm_(self.probe_model.parameters(), grad_clip),
            }
        else:
            grad_metrics = {
                'grad_norm': nn.utils.clip_grad_norm_(self.parameters(), grad_clip),
            }
        return grad_metrics

    def init_state(self, batch_size: int):
        return self.wm.init_state(batch_size)

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      imag_horizon: Optional[int] = None,
                      do_open_loop=False,
                      do_image_pred=False,
                      do_dream_tensors=False,