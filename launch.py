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
    obs_space,act_space= create_env(conf.env_id, conf.env_no_terminal, conf.