
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributions as D

from ..math_functions import *
from .common import *
from .. import tools_v3


class MultiDecoder(nn.Module):

    def __init__(self, features_dim, conf):
        super().__init__()
        self.image_weight = conf.image_weight
        self.vecobs_weight = conf.vecobs_weight
        self.reward_weight = conf.reward_weight
        self.terminal_weight = conf.terminal_weight

        if conf.image_decoder == 'cnn':
            self.image = ConvDecoder(in_dim=features_dim,
                                     out_channels=conf.image_channels,
                                     cnn_depth=conf.cnn_depth)
        elif conf.image_decoder == 'dense':
            self.image = CatImageDecoder(in_dim=features_dim,
                                         out_shape=(conf.image_channels, conf.image_size, conf.image_size),
                                         hidden_layers=conf.image_decoder_layers,
                                         layer_norm=conf.layer_norm,
                                         min_prob=conf.image_decoder_min_prob)
        elif not conf.image_decoder:
            self.image = None
        else:
            assert False, conf.image_decoder
        if conf.wm_type=='v2':
            if conf.reward_decoder_categorical:
                self.reward = DenseCategoricalSupportDecoder(
                    in_dim=features_dim,
                    support=clip_rewards_np(conf.reward_decoder_categorical, conf.clip_rewards),  # reward_decoder_categorical values are untransformed 
                    hidden_layers=conf.reward_decoder_layers,
                    layer_norm=conf.layer_norm)
            else:
                self.reward = DenseNormalDecoder(in_dim=features_dim, hidden_layers=conf.reward_decoder_layers, layer_norm=conf.layer_norm)

            self.terminal = DenseBernoulliDecoder(in_dim=features_dim, hidden_layers=conf.terminal_decoder_layers, layer_norm=conf.layer_norm)
        elif conf.wm_type=='v3':
            if conf.reward_head == "symlog_disc":