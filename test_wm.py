
import argparse
import json
import os
import time
from logging import info
from distutils.util import strtobool
from multiprocessing import Process
from typing import List
import torch
from pydreamer.models import Dreamer_agent

import generator
import train
from pydreamer.tools import (configure_logging, mlflow_log_params,
                             mlflow_init, print_once, read_yamls)
import time
from collections import defaultdict
from datetime import datetime
from logging import critical, debug, error, info, warning
from typing import Iterator, Optional

import mlflow
import numpy as np
import scipy.special
import torch
import numpy as np
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader

from pydreamer import tools
from pydreamer.data import DataSequential, MlflowEpisodeRepository
from pydreamer.models import *
from pydreamer.models.math_functions import map_structure, nanmean
from pydreamer.preprocessing import Preprocessor, WorkerInfoPreprocess
from pydreamer.tools import *
from make_gif import make_gif_wm

def make_args():
    parser = argparse.ArgumentParser(description="argument parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--artifact_uri", required=False, type=str, default="/home/chenghan/pydreamer/mlruns/0/c1d429acbbff43afb0e2edd18d11ebce/artifacts")
    parser.add_argument("--action_type", default='fixed_online', type=str)
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument('--configs', nargs='+', required=True)

    args, remaining = parser.parse_known_args()
    # print(args)
    # print(remaining)
    conf = {
        "artifact_uri": args.artifact_uri,
        "action_type": args.action_type,
        "index": args.index,
    } ##所有参数的集合
    configs = read_yamls('./config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    # Override config from command-line，覆盖掉yaml中的设置

    parser = argparse.ArgumentParser(description="argument parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--configs', nargs='+', required=True)
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = lambda x: bool(strtobool(x))
        parser.add_argument(f'--{key}', type=type_, default=value)
    
    conf = parser.parse_args(remaining)

    return conf



def prepare_batch_npz(data: Dict[str, Tensor], take_b=999):

    def unpreprocess(key: str, val: Tensor) -> np.ndarray:
        if take_b < val.shape[1]:
            val = val[:, :take_b]

        x = val.detach().cpu().numpy()  # (T,B,*)
        if x.dtype in [np.float16, np.float64]:
            x = x.astype(np.float32)

        if len(x.shape) == 2:  # Scalar
            pass

        elif len(x.shape) == 3:  # 1D vector
            pass

        elif len(x.shape) == 4:  # 2D tensor
            pass

        elif len(x.shape) == 5:  # 3D tensor - image
            assert x.dtype == np.float32 and (key.startswith('image') or key.startswith('map')), \
                f'Unexpected 3D tensor: {key}: {x.shape}, {x.dtype}'

            if x.shape[-1] == x.shape[-2]:  # (T,B,C,W,W)
                x = x.transpose(0, 1, 3, 4, 2)  # => (T,B,W,W,C)
            assert x.shape[-2] == x.shape[-3], 'Assuming rectangular images, otherwise need to improve logic'

            if x.shape[-1] in [1, 3]:
                # RGB or grayscale
                x = ((x + 0.5) * 255.0).clip(0, 255).astype('uint8')
            elif np.allclose(x.sum(axis=-1), 1.0) and np.allclose(x.max(axis=-1), 1.0):
                # One-hot
                x = x.argmax(axis=-1)
            else:
                # Categorical logits
                assert key in ['map_rec', 'image_rec', 'image_pred'], \
                    f'Unexpected 3D categorical logits: {key}: {x.shape}'
                x = scipy.special.softmax(x, axis=-1)

        x = x.swapaxes(0, 1)  # type: ignore  # (T,B,*) => (B,T,*)
        return x

    return {k: unpreprocess(k, v) for k, v in data.items()}

# def test_dream(model,obs,states,action_type):
#     with torch.no_grad():  # careful not to invoke modules first time under no_grad (https://github.com/pytorch/pytorch/issues/60164)
#         # The reason we don't just take real features_dream is because it's really big (H*T*B*I),
#         # and here for inspection purposes we only dream from first step, so it's (H*B).
#         # Oh, and we set here H=T-1, so we get (T,B), and the dreamed experience aligns with actual.
#         # 这里实际做的时候，T=1，只从第一步想象
#         in_state_dream=map_structure(states, lambda x: x.detach()[0, :, 0])  # type: ignore  # (T,B,I) => (B)
#         ## 基本上只改了这一步
#         # non_zero_indices = torch.nonzero(in_state_dream[1])
#         if action_type=='fixed_online':
#             features_dream, actions_dream, rewards_dream, terminals_dream = model.dream_cond_action(in_state_dream, obs['action'])
#             image_dream = model.wm.decoder.image.forward(features_dream)
#             dream_tensors = dict(action_pred=actions_dream,  # first action is real from previous step
#                                     reward_pred=rewards_dream.mean,
#                                     terminal_pred=terminals_dream.mean,
#                                     image_pred=image_dream,)

#         elif action_type=='adapt_online':
#             features_dream, actions_dream, rewards_dream, terminals_dream = model.dream(in_state_dream, obs['action'].shape[0] - 1)  # H = T-1
#             image_dream = model.wm.decoder.image.forward(features_dream)
#             dream_tensors = dict(action_pred=torch.cat([obs['action'][:1], actions_dream]),  # first action is real from previous step
#                              reward_pred=rewards_dream.mean,
#                              terminal_pred=terminals_dream.mean,
#                              image_pred=image_dream,
#                              )
#         elif action_type=='disturb_online':
#             features_dream, actions_dream, rewards_dream, terminals_dream = model.dream(in_state_dream, obs['action'].shape[0] - 1,perturb='1')  # H = T-1
#             image_dream = model.wm.decoder.image.forward(features_dream)
#             dream_tensors = dict(action_pred=torch.cat([obs['action'][:1], actions_dream]),  # first action is real from previous step
#                              reward_pred=rewards_dream.mean,
#                              terminal_pred=terminals_dream.mean,
#                              image_pred=image_dream,
#                              )
#         elif action_type=='offline':
#             features_dream, actions_dream, rewards_dream, terminals_dream = model.dream(in_state_dream, obs['action'].shape[0] - 1,perturb='2')  # H = T-1
#             image_dream = model.wm.decoder.image.forward(features_dream)
#             dream_tensors = dict(action_pred=torch.cat([obs['action'][:1], actions_dream]),  # first action is real from previous step
#                              reward_pred=rewards_dream.mean,
#                              terminal_pred=terminals_dream.mean,
#                              image_pred=image_dream,
#                              )
#         assert dream_tensors['action_pred'].shape == obs['action'].shape
#         assert dream_tensors['image_pred'].shape == obs['image'].shape