
import io
import logging
import os
import posixpath
import sys
import tempfile
import time
import warnings
from logging import debug, info, exception
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import yaml

try:
    from mlflow.store.artifact.artifact_repo import ArtifactRepository
except:
    ArtifactRepository = Any  # just for type annotation

# Ignore Google Cloud Storage warnings
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

print_once_keys = set()


def print_once(key, obj):
    if key not in print_once_keys:
        print_once_keys.add(key)
        logging.debug(f'{key} {obj}')


def to_list(s):
    return s if isinstance(s, list) else [s]


def read_yamls(dir):
    conf = {}
    no_conf = True
    for config_file in Path(dir).glob('**/*.yaml'):
        no_conf = False
        with config_file.open('r') as f:
            conf.update(yaml.safe_load(f))
    if no_conf:
        print(f'WARNING: No yaml files found in {dir}')
    return conf


def mlflow_init(wait_for_resume=False):
    import mlflow
    run_name = os.environ.get('MLFLOW_RUN_NAME')
    resume_id = os.environ.get('MLFLOW_RESUME_ID')
    uri = os.environ.get('MLFLOW_TRACKING_URI', 'local')
    # print(run_name,resume_id,uri)

    run = mlflow.active_run()
    if run:
        # Run already active
        pass

    elif os.environ.get('MLFLOW_RUN_ID'):
        # Run not active, but specific ID set (probably subprocess)
        run = mlflow.start_run(run_id=os.environ['MLFLOW_RUN_ID'])
        info(f'Reinitialized mlflow run {run.info.run_id} ({resume_id}) in {uri}/{run.info.experiment_id}')

    else:
        resume_run_id = None
        if resume_id:
            # Resume ID specified - try to find the same run
            while True:
                runs = mlflow.search_runs(filter_string=f'tags.resume_id="{resume_id}"')
                if len(runs) > 0:
                    resume_run_id = runs.run_id.iloc[0]  # type: ignore
                    break
                else:
                    if wait_for_resume:
                        debug(f'Waiting until mlflow run ({resume_id}) is available...')
                        time.sleep(10)
                    else:
                        break
        else:
            assert not wait_for_resume, "Wait for resume, but no MLFLOW_RESUME_ID"

        if resume_run_id:
            # Resuming run
            run = mlflow.start_run(run_id=resume_run_id)
            info(f'Resumed mlflow run {run.info.run_id} ({resume_id}) in {uri}/{run.info.experiment_id}')
        else:
            # Starting new run
            mlflow.set_tracking_uri("mlruns")
            # mlflow.create_experiment("your_experiment_name", "1")
            # experiment_id = mlflow.get_experiment_by_name('1').experiment_id
            run = mlflow.start_run(run_name=run_name, tags={'resume_id': resume_id or ''})
            print(run.info.run_id)
            info(f'Started mlflow run {run.info.run_id} ({resume_id}) in {uri}/{run.info.experiment_id}')

    os.environ['MLFLOW_RUN_ID'] = run.info.run_id  # for subprocesses
    return run


def mlflow_log_params(params: dict):
    import mlflow
    MAX_VALUE_LENGTH = 250
    MAX_BATCH_SIZE = 100
    kvs = [