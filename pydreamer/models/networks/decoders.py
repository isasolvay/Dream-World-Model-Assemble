
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
                self.reward = MLP_v3(
                    features_dim,  # pytorch version
                    (255,),
                    conf.reward_layers,
                    conf.units,
                    conf.act,
                    conf.norm,
                    dist=conf.reward_head,
                    outscale=0.0,
                    device=conf.device,
                )
            else:
                self.reward = MLP_v3(
                    features_dim,  # pytorch version
                    [],
                    conf.reward_layers,
                    conf.units,
                    conf.act,
                    conf.norm,
                    dist=conf.reward_head,
                    outscale=0.0,
                    device=conf.device,
                )
            self.terminal = MLP_v3(
                features_dim,  # pytorch version
                [],
                conf.terminal_layers,
                conf.units,
                conf.act,
                conf.norm,
                dist="binary",
                device=conf.device,
            )

        if conf.vecobs_size:
            self.vecobs = DenseNormalDecoder(in_dim=features_dim, out_dim=conf.vecobs_size, hidden_layers=4, layer_norm=conf.layer_norm)
        else:
            self.vecobs = None

    def training_step(self,
                      features: TensorTBIF,
                      obs: Dict[str, Tensor],
                      extra_metrics: bool = False
                      ) -> Tuple[TensorTBI, Dict[str, Tensor], Dict[str, Tensor]]:
        tensors = {}
        metrics = {}
        loss_reconstr = 0

        if self.image:
            loss_image_tbi, loss_image, image_rec = self.image.training_step(features, obs['image'])
            loss_reconstr += self.image_weight * loss_image_tbi
            metrics.update(loss_image=loss_image.detach().mean())
            tensors.update(loss_image=loss_image.detach(),
                           image_rec=image_rec.detach())

        if self.vecobs:
            loss_vecobs_tbi, loss_vecobs, vecobs_rec = self.vecobs.training_step(features, obs['vecobs'])
            loss_reconstr += self.vecobs_weight * loss_vecobs_tbi
            metrics.update(loss_vecobs=loss_vecobs.detach().mean())
            tensors.update(loss_vecobs=loss_vecobs.detach(),
                           vecobs_rec=vecobs_rec.detach())

        loss_reward_tbi, loss_reward, reward_rec = self.reward.training_step(features, obs['reward'])
        loss_reconstr += self.reward_weight * loss_reward_tbi
        metrics.update(loss_reward=loss_reward.detach().mean())
        tensors.update(loss_reward=loss_reward.detach(),
                       reward_rec=reward_rec.detach())

        loss_terminal_tbi, loss_terminal, terminal_rec = self.terminal.training_step(features, obs['terminal'])
        loss_reconstr += self.terminal_weight * loss_terminal_tbi
        metrics.update(loss_terminal=loss_terminal.detach().mean())
        tensors.update(loss_terminal=loss_terminal.detach(),
                       terminal_rec=terminal_rec.detach())

        if extra_metrics:
            if isinstance(self.reward, DenseCategoricalSupportDecoder):
                # TODO: logic should be moved to appropriate decoder
                reward_cat = self.reward.to_categorical(obs['reward'])
                for i in range(len(self.reward.support)):
                    # Logprobs for specific categorical reward values
                    mask_rewardp = reward_cat == i  # mask where categorical reward has specific value
                    loss_rewardp = loss_reward * mask_rewardp / mask_rewardp  # set to nan where ~mask
                    metrics[f'loss_reward{i}'] = nanmean(loss_rewardp)  # index by support bucket, not by value
                    tensors[f'loss_reward{i}'] = loss_rewardp
            else:
                for sig in [-1, 1]:
                    # Logprobs for positive and negative rewards
                    mask_rewardp = torch.sign(obs['reward']) == sig  # mask where reward is positive or negative
                    loss_rewardp = loss_reward * mask_rewardp / mask_rewardp  # set to nan where ~mask
                    metrics[f'loss_reward{sig}'] = nanmean(loss_rewardp)
                    tensors[f'loss_reward{sig}'] = loss_rewardp

            mask_terminal1 = obs['terminal'] > 0  # mask where terminal is 1
            loss_terminal1 = loss_terminal * mask_terminal1 / mask_terminal1  # set to nan where ~mask
            metrics['loss_terminal1'] = nanmean(loss_terminal1)
            tensors['loss_terminal1'] = loss_terminal1

        return loss_reconstr, metrics, tensors
class MultiDecoder_v3(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
    ):
        super().__init__()
        ## image decoder part
        excluded = ("reset", "is_last", "terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}