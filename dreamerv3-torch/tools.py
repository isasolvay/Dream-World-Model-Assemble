
import datetime
import collections
import io
import os
import json
import pathlib
import re
import time
import uuid

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter


to_np = lambda x: x.detach().cpu().numpy()


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class TimeRecording:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class Logger:
    def __init__(self, logdir, step):
        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False, step=False):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            self._writer.add_scalar("scalars/" + name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)

        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        self._writer.add_scalar("scalars/" + name, value, step)

    def offline_video(self, name, value, step):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        self._writer.add_video(name, value, step, 16)


def simulate(
    agent,
    envs,
    cache,
    directory,
    logger,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
):
    # initialize or unpack simulation state
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
        reward = [0] * len(envs)
    else:
        step, episode, done, length, obs, agent_state, reward = state
    while (steps and step < steps) or (episodes and episode < episodes):
        # reset envs if necessary
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            results = [envs[i].reset() for i in indices]
            results = [r() for r in results]
            for index, result in zip(indices, results):
                t = result.copy()
                t = {k: convert(v) for k, v in t.items()}
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, envs[index].id, t)
                # replace obs with done by initial state
                obs[index] = result
        # step agents
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        action, agent_state = agent(obs, done, agent_state)
        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(envs)
        # step envs
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += len(envs)
        length *= 1 - done
        # add to cache
        for a, result, env in zip(action, results, envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            add_to_cache(cache, env.id, transition)

        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            # logging for done episode
            for i in indices:
                save_episodes(directory, {envs[i].id: cache[envs[i].id]})
                length = len(cache[envs[i].id]["reward"]) - 1
                score = float(np.array(cache[envs[i].id]["reward"]).sum())
                video = cache[envs[i].id]["image"]
                if not is_eval:
                    step_in_dataset = erase_over_episodes(cache, limit)
                    logger.scalar(f"dataset_size", step_in_dataset)
                    logger.scalar(f"train_return", score)
                    logger.scalar(f"train_length", length)
                    logger.scalar(f"train_episodes", len(cache))
                    logger.write(step=logger.step)
                else:
                    if not "eval_lengths" in locals():
                        eval_lengths = []
                        eval_scores = []
                        eval_done = False
                    # start counting scores for evaluation
                    eval_scores.append(score)
                    eval_lengths.append(length)

                    score = sum(eval_scores) / len(eval_scores)
                    length = sum(eval_lengths) / len(eval_lengths)
                    logger.video(f"eval_policy", np.array(video)[None])

                    if len(eval_scores) >= episodes and not eval_done:
                        logger.scalar(f"eval_return", score)