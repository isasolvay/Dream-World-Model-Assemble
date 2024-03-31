
import time
from collections import defaultdict
from datetime import datetime
from logging import critical, debug, error, info, warning
from typing import Iterator, Optional

import mlflow
import numpy as np
import scipy.special
import torch
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
from pydreamer.models import tools_v3

def run(conf,space):
    configure_logging(prefix='[TRAIN]')
    mlrun = mlflow_init()
    artifact_uri = mlrun.info.artifact_uri

    torch.distributions.Distribution.set_default_validate_args(False)
    torch.backends.cudnn.benchmark = True  # type: ignore
    device = torch.device(conf.device)
    
    #Basic env
    obs_space=space["obs"]
    act_space=space["act"]
    
    # Data directories

    if conf.offline_data_dir:
        online_data = False
        input_dirs = to_list(conf.offline_data_dir)
    else:
        online_data = True
        input_dirs = [
            f'{artifact_uri}/episodes/{i}'
            for i in range(max(conf.generator_workers_train, conf.generator_workers))
        ]
    
    if conf.offline_prefill_dir:
        input_dirs.extend(to_list(conf.offline_prefill_dir))

    if conf.offline_eval_dir:
        eval_dirs = to_list(conf.offline_eval_dir)
    else:
        eval_dirs = [
            f'{artifact_uri}/episodes_eval/{i}'
            for i in range(max(conf.generator_workers_eval, conf.generator_workers))
        ]

    if conf.offline_test_dir:
        test_dirs = to_list(conf.offline_test_dir)
    else:
        test_dirs = eval_dirs

    # Wait for prefill

    if online_data:
        while True:
            data_train_stats = DataSequential(MlflowEpisodeRepository(input_dirs), conf.batch_length, conf.batch_size, check_nonempty=False)
            mlflow_log_metrics({
                'train/data_steps': data_train_stats.stats_steps,
                'train/data_env_steps': data_train_stats.stats_steps * conf.env_action_repeat,
                '_timestamp': datetime.now().timestamp(),
            }, step=0)
            if data_train_stats.stats_steps < conf.generator_prefill_steps:
                debug(f'Waiting for prefill: {data_train_stats.stats_steps}/{conf.generator_prefill_steps} steps...')
                time.sleep(10)
            else:
                info(f'Done prefilling: {data_train_stats.stats_steps}/{conf.generator_prefill_steps} steps.')
                break

        if data_train_stats.stats_steps * conf.env_action_repeat >= conf.n_env_steps:
            # Prefill-only job, or resumed already finished job
            info(f'Finished {conf.n_env_steps} env steps.')
            return

    # Data reader
    amp=True if conf.precision == 16 else False
    data = DataSequential(MlflowEpisodeRepository(input_dirs),
                          conf.batch_length,
                          conf.batch_size,
                          skip_first=True,
                          reload_interval=120 if online_data else 0,
                          buffer_size=conf.buffer_size if online_data else conf.buffer_size_offline,
                          reset_interval=conf.reset_interval,
                          allow_mid_reset=conf.allow_mid_reset)
    preprocess = Preprocessor(image_categorical=conf.image_channels if conf.image_categorical else None,
                              image_key=conf.image_key,
                              map_categorical=conf.map_channels if conf.map_categorical else None,
                              map_key=conf.map_key,
                              action_dim=conf.action_dim,
                              clip_rewards=conf.clip_rewards,
                              amp=amp and device.type == 'cuda')

    # MODEL

    if conf.model == 'dreamer':
        model = Dreamer_agent(conf,obs_space,act_space,data_train_stats.stats_steps)
    else:
        model: Dreamer_agent = WorldModelProbe(conf)  # type: ignore
    model.to(device)
    print(model)
    # print(repr(model))
    mlflow_log_text(repr(model), 'architecture.txt')

    optimizers = model.init_optimizers(conf.adam_lr, conf.adam_lr_actor, conf.adam_lr_critic, conf.adam_eps,conf.adam_ac_eps)
    resume_step = tools.mlflow_load_checkpoint(model, optimizers)
    if resume_step:
        info(f'Loaded model from checkpoint epoch {resume_step}')
