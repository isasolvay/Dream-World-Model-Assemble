
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributions as D

from .. math_functions import *
from .common import *
from .. import tools_v3


class MultiEncoder_v2(nn.Module):

    def __init__(self, shapes,conf):
        super().__init__()
        self.wm_type=conf.wm_type
        self.reward_input = conf.reward_input
        

        # if conf.image_encoder == 'cnn':
        #     self.encoder_image = ConvEncoder(in_channels=encoder_channels,
        #                                      cnn_depth=conf.cnn_depth)
        # elif conf.image_encoder == 'dense':
        #     if conf.wm_type=='v2':
        #         self.encoder_image = DenseEncoder(in_dim=conf.image_size * conf.image_size * encoder_channels,
        #                                       out_dim=256,
        #                                       hidden_layers=conf.image_encoder_layers,
        #                                       layer_norm=conf.layer_norm)
        # elif not conf.image_encoder:
        #     self.encoder_image = None
        # else:
        #     assert False, conf.image_encoder
            
        #     # vecons_size=0
        # if conf.vecobs_size:
        #     self.encoder_vecobs = MLP_v2(conf.vecobs_size, 256, hidden_dim=400, hidden_layers=2, layer_norm=conf.layer_norm)
        # else:
        #     self.encoder_vecobs = None

        # assert self.encoder_image or self.encoder_vecobs, "Either image_encoder or vecobs_size should be set"
        # self.out_dim = ((self.encoder_image.out_dim if self.encoder_image else 0) +
        #                 (self.encoder_vecobs.out_dim if self.encoder_vecobs else 0))
        
        excluded = ("reset", "is_last", "terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(conf.encoder["cnn_keys"], k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(conf.encoder["mlp_keys"], k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)
        
        if conf.image_encoder == 'cnn':
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            if conf.reward_input:
                    input_ch+=2  # + reward, terminal
            # else:
            #         encoder_channels = conf.image_channels
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self.encoder_image = ConvEncoder(
                input_shape, self.wm_type,conf.encoder["cnn_depth"], conf.encoder["act"], conf.encoder["norm"], conf.encoder["kernel_size"], 
                conf.encoder["minres"]
            )
            # self.encoder_image = ConvEncoder(in_channels=encoder_channels,
            #                                  cnn_depth=conf.cnn_depth)
        
        elif conf.image_encoder == 'dense':
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self.encoder_image = DenseEncoder(in_dim=input_size,
                                            out_dim=256,
                                            hidden_layers=conf.image_encoder_layers,
                                            layer_norm=conf.layer_norm)
        elif not conf.image_encoder:
            self.encoder_image = None
        else:
            assert False, conf.image_encoder
            
            # vecons_size=0
        if conf.vecobs_size:
            self.encoder_vecobs = MLP_v2(conf.vecobs_size, 256, hidden_dim=400, hidden_layers=2, layer_norm=conf.layer_norm)
        else:
            self.encoder_vecobs = None

        assert self.encoder_image or self.encoder_vecobs, "Either image_encoder or vecobs_size should be set"
        self.out_dim = ((self.encoder_image.out_dim if self.encoder_image else 0) +
                        (self.encoder_vecobs.out_dim if self.encoder_vecobs else 0))
        

    def forward(self, obs: Dict[str, Tensor]) -> TensorTBE:
        # TODO:
        #  1) Make this more generic, e.g. working without image input or without vecobs
        #  2) Treat all inputs equally, adding everything via linear layer to embed_dim

        embeds = []

        if self.encoder_image:
            image = obs['image']
            T, B, C, H, W = image.shape