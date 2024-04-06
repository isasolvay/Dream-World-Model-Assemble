
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

    # ---------------------
    # TRAINING
    # ---------------------

    start_time = time.time()
    steps = resume_step or 0
    last_time = start_time
    last_steps = steps
    metrics = defaultdict(list)
    metrics_max = defaultdict(list)

    timers = {}

    def timer(name, verbose=False):
        if name not in timers:
            timers[name] = Timer(name, verbose)
        return timers[name]

    states = {}  # by worker
    data_iter = iter(DataLoader(WorkerInfoPreprocess(preprocess(data)),
                                batch_size=None,
                                num_workers=conf.data_workers,
                                prefetch_factor=20 if conf.data_workers else 2,  # GCS download has to be shorter than this many batches (e.g. 1sec < 20*300ms)
                                pin_memory=True))

    scaler = GradScaler(enabled=amp)

    with get_profiler(conf) as profiler:
        while True:
            with timer('total'):
                profiler.step()
                steps += 1
                will_log_batch = steps % conf.logbatch_interval == 1
                will_image_pred = (
                    will_log_batch or
                    steps % conf.log_interval >= int(conf.log_interval * 0.9)  # 10% of batches
                )

                # Make batch

                with timer('data'):

                    batch, wid = next(data_iter)
                    obs: Dict[str, Tensor] = map_structure(batch, lambda x: x.to(device))  # type: ignore

                # Forward

                with timer('forward'):
                    with autocast(enabled=amp):

                        state = states.get(wid)
                        if state is None:
                            state = model.init_state(conf.batch_size * conf.iwae_samples)
                        losses, new_state, loss_metrics, tensors, dream_tensors = \
                            model.training_step(
                                obs,
                                state,
                                do_image_pred=will_image_pred,
                                do_dream_tensors=will_log_batch)
                        if conf.keep_state:
                            states[wid] = new_state

                # Backward

                with timer('backward'):

                    for opt in optimizers:
                        opt.zero_grad()
                    for loss in losses:
                        scaler.scale(loss).backward()  # type: ignore

                # Grad step

                with timer('gradstep'):  # CUDA wait happens here

                    for opt in optimizers:
                        scaler.unscale_(opt)
                    grad_metrics = model.grad_clip(conf.grad_clip, conf.grad_clip_ac)
                    for opt in optimizers:
                        scaler.step(opt)
                    scaler.update()
                    
                    # for name, param in model.ac.critic.named_parameters():
                    #     if param.grad is not None:
                    #         print(name, param.grad.norm().item())
                    #     else:
                    #         print(param.grad)
                    # for name, param in model.ac.actor.named_parameters():
                    #     if param.grad is not None:
                    #         print(name, param.grad.norm().item())
                    #     else:
                    #         print(param.grad)
                    # for name, param in model.wm.dynamics.named_parameters():
                    #     if param.grad is not None:
                    #         print(name, param.grad.norm().item())
                    #     else:
                    #         print(param.grad)

                with timer('other'):

                    # Metrics

                    for k, v in loss_metrics.items():
                        # print(k)
                        if isinstance(v, (np.ndarray, torch.Tensor)):
                            # 如果 v 是数组或张量，则提取单个值
                            value = v.item()
                        else:
                            # 否则，假设 v 已经是一个数值
                            value = v
                        if not np.isnan(value):
                            metrics[k].append(value)
                    for k, v in grad_metrics.items():
                        if np.isfinite(v.item()):  # It's ok for grad norm to be inf, when using amp
                            metrics[k].append(v.item())
                            metrics_max[k].append(v.item())
                    for k in ['reward', 'reset', 'terminal']:
                        metrics[f'data_{k}'].append(batch[k].float().mean().item())
                    for k in ['reward']:
                        metrics_max[f'data_{k}'].append(batch[k].max().item())

                    # Log sample

                    if will_log_batch:
                        log_batch_npz(batch, tensors, f'{steps:07}.npz', subdir='d2_wm_closed')
                    if dream_tensors:
                        log_batch_npz(batch, dream_tensors, f'{steps:07}.npz', subdir='d2_wm_dream')

                    # Log data buffer size

                    if online_data and steps % conf.logbatch_interval == 0:
                        data_train_stats = DataSequential(MlflowEpisodeRepository(input_dirs), conf.batch_length, conf.batch_size)
                        metrics['data_steps'].append(data_train_stats.stats_steps)
                        metrics['data_env_steps'].append(data_train_stats.stats_steps * conf.env_action_repeat)
                        while steps*200/6>data_train_stats.stats_steps:
                            time.sleep(100)
                            data_train_stats = DataSequential(MlflowEpisodeRepository(input_dirs), conf.batch_length, conf.batch_size)
                        if data_train_stats.stats_steps * conf.env_action_repeat >= conf.n_env_steps:
                            info(f'Finished {conf.n_env_steps} env steps.')
                            return

                    # Log metrics

                    if steps % conf.log_interval == 0:
                        metrics = {f'train/{k}': np.array(v).mean() for k, v in metrics.items()}
                        metrics.update({f'train/{k}_max': np.array(v).max() for k, v in metrics_max.items()})
                        metrics['train/steps'] = steps
                        metrics['_step'] = steps
                        # metrics['_loss'] = metrics.get('train/loss_model', 0)
                        metrics['_losses'] =sum(losses).item()
                        metrics['_timestamp'] = datetime.now().timestamp()

                        t = time.time()