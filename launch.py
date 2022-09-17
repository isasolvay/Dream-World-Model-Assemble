import argparse
import json
import os
import time
from logging import info
from distutils.util import strtobool
from multiprocessing import Process
from typing import List
from pydreamer.tools import *
from pydreamer.data import DataSequential, MlflowEpisodeRepository
import generator
import train
# from pydreamer.models import dreamer
from pydreamer.tools import (configure_logging, mlflow_log_params,mlflow_log_text,
                             mlflow_init, print_once, read_yamls)
from pydreamer import envs
from pydreamer.envs.__init__ import create_env

os.environ["MUJOCO_GL"]='egl'
def launch():
    configure_logging('[launcher]')
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()

    # Config from YAML

    conf = {}
    configs = read_yamls('./config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    # Override config from command-line

    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = lambda x: bool(strtobool(x))
        parser.add_argument(f'--{key}', type=type_, default=value)
    conf = parser.parse_args(remaining)

    # Mlflow

    worker_type, worker_index = get_worker_info()
    is_main_worker = worker_type is None or worker_type == 'learner'
    mlrun = mlflow_init(wait_for_resume=not is_main_worker)
    artifact_uri = mlrun.info.artifact_uri
    mlflow_log_params(vars(conf))
    
    # What env do you want? Basic env-info
    obs_space,act_space= create_env(conf.env_id, conf.env_no_terminal, conf.env_time_limit, conf.env_action_repeat, 0,conf=conf,info_only=True)
    space={}
    space["obs"]=obs_space
    space["act"]=act_space
    
    # To know the num of steps, generator should know Data directories

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

    # if conf.offline_eval_dir:
    #     eval_dirs = to_list(conf.offline_eval_dir)
    # else:
    #     eval_dirs = [
    #         f'{artifact_uri}/episodes_eval/{i}'
    #         for i in range(max(conf.generator_workers_eval, conf.generator_workers))
    #     ]

    # if conf.offline_test_dir:
    #     test_dirs = to_list(conf.offline_test_dir)
    # else:
    #     test_dirs = eval_dirs
        
    
    
    # Launch train+eval generators
    ## I don't know why, but we should do it by hand
    # conf.generator_workers=1
    subprocesses: List[Process] = []
    for i in range(conf.generator_workers):
        print('--------generator number--------------')
        print(conf.generator_workers)
        if belongs_to_worker('generator', i):
            info(f'Launching train+eval generator {i}')
            p = launch_generator(
                conf.env_id,
                conf,
                save_uri=f'{artifact_uri}/episodes/{i}',
                save_uri2=f'{artifact_uri}/episodes_eval/{i}',
                num_steps=conf.n_env_steps // conf.env_action_repeat // conf.generator_workers,
                limit_step_ratio=conf.limit_step_ratio / conf.generator_workers,
                worker_id=i,
                policy_main='network',
                policy_prefill=conf.generator_prefill_policy,
                num_steps_prefill=conf.generator_prefill_steps // conf.generator_workers,
                split_fraction=0.05,
                input_dirs=input_dirs,
            )
            subprocesses.append(p)

    # Launch train generators

    for i in range(conf.generator_workers_train):
        if belongs_to_worker('generator_train', i):
            info(f'Launching train generator {i}')
            p = launch_generator(
                conf.env_id,
                conf,
                f'{artifact_uri}/episodes/{i}',
                num_steps=conf.n_env_steps // conf.env_action_repeat // conf.generator_workers,
                lim