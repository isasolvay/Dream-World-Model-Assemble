
import argparse
import logging
import logging.config
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import chain
from logging import critical, debug, error, info, warning
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.distributions as D

from pydreamer.data import DataSequential,MlflowEpisodeRepository
from pydreamer.envs import create_env
from pydreamer.models import *
from pydreamer.models.math_functions import map_structure
from pydreamer.preprocessing import Preprocessor
from pydreamer.tools import *


def main(conf,
    env_id='MiniGrid-MazeS11N-v0',
         save_uri=None,
         save_uri2=None,
         worker_id=0,
         policy_main='random',
         policy_prefill='random',
         num_steps=int(1e6),
         num_steps_prefill=0,
         env_no_terminal=False,
         env_time_limit=0,
         env_action_repeat=1,
         limit_step_ratio=0.0,
         steps_per_npz=1000,
         model_reload_interval=120,
         model_conf=dict(),
         log_mlflow_metrics=True,
         split_fraction=0.0,
         metrics_prefix='agent',
         metrics_gamma=0.99,
         log_every=10,
         input_dirs=None
         ):

    configure_logging(prefix=f'[GEN {worker_id}]', info_color=LogColorFormatter.GREEN)
    mlrun = mlflow_init()
    info(f'Generator {worker_id} started:'
         f' env={env_id}'
         f', n_steps={num_steps:,}'
         f', n_prefill={num_steps_prefill:,}'
         f', split_fraction={split_fraction}'
         f', metrics={metrics_prefix if log_mlflow_metrics else None}'
         f', save_uri={save_uri}')

    if not save_uri:
        save_uri = f'{mlrun.info.artifact_uri}/episodes/{worker_id}'
    if not save_uri2:
        assert split_fraction == 0.0, 'Specify two save destinations, if splitting'

    repository = MlflowEpisodeRepository(save_uri)
    repository2 = MlflowEpisodeRepository(save_uri2) if save_uri2 else repository
    nfiles, steps_saved, episodes = repository.count_steps()
    info(f'Found existing {nfiles} files, {episodes} episodes, {steps_saved} steps in {repository}')

    # Env

    env = create_env(env_id, env_no_terminal, env_time_limit, env_action_repeat, worker_id,conf)

    # Policy

    if num_steps_prefill:
        # Start with prefill policy
        info(f'Prefill policy: {policy_prefill}')
        policy = create_policy(policy_prefill, env, model_conf,input_dirs)
        is_prefill_policy = True
    else:
        info(f'Policy: {policy_main}')
        policy = create_policy(policy_main, env, model_conf,input_dirs)
        is_prefill_policy = False

    # RUN

    datas = []
    last_model_load = 0
    model_step = 0
    metrics_agg = defaultdict(list)
    all_returns = []
    steps = 0

    while steps_saved < num_steps:

        # Switch policy prefill => main

        if is_prefill_policy and steps_saved >= num_steps_prefill:
            info(f'Switching to main policy: {policy_main}')
            policy = create_policy(policy_main, env, model_conf,input_dirs)
            is_prefill_policy = False

        # Load network

        if isinstance(policy, NetworkPolicy):
            if time.time() - last_model_load > model_reload_interval:
                while True:
                    # takes ~10sec to load checkpoint
                    model_step = mlflow_load_checkpoint(policy.model, map_location='cpu')  # type: ignore
                    if model_step:
                        info(f'Generator loaded model checkpoint {model_step}')
                        last_model_load = time.time()
                        break
                    else:
                        debug('Generator model checkpoint not found, waiting...')
                        time.sleep(10)

            if limit_step_ratio and steps_saved >= model_step * limit_step_ratio:
                # Rate limiting - keep looping until new model checkpoint is loaded
                time.sleep(1)
                continue

        # Unroll one episode

        epsteps = 0