
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
                        logger.scalar(f"eval_length", length)
                        logger.scalar(f"eval_episodes", len(eval_scores))
                        logger.write(step=logger.step)
                        eval_done = True
    if is_eval:
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)


class CollectDataset:
    def __init__(
        self, env, mode, train_eps, eval_eps=dict(), callbacks=None, precision=32
    ):
        self._env = env
        self._callbacks = callbacks or ()
        self._precision = precision
        self._episode = None
        self._cache = dict(train=train_eps, eval=eval_eps)[mode]
        self._temp_name = str(uuid.uuid4())

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {k: self._convert(v) for k, v in obs.items()}
        transition = obs.copy()
        if isinstance(action, dict):
            transition.update(action)
        else:
            transition["action"] = action
        transition["reward"] = reward
        transition["discount"] = info.get("discount", np.array(1 - float(done)))
        self._episode.append(transition)
        self.add_to_cache(transition)
        if done:
            # detele transitions before whole episode is stored
            del self._cache[self._temp_name]
            self._temp_name = str(uuid.uuid4())
            for key, value in self._episode[1].items():
                if key not in self._episode[0]:
                    self._episode[0][key] = 0 * value
            episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            info["episode"] = episode
            for callback in self._callbacks:
                callback(episode)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        # missing keys will be filled with a zeroed out version of the first
        # transition, because we do not know what action information the agent will
        # pass yet.
        transition["reward"] = 0.0
        transition["discount"] = 1.0
        self._episode = [transition]
        self.add_to_cache(transition)
        return obs


def add_to_cache(cache, id, transition):
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size):
    step_in_dataset = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
            not dataset_size
            or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    return step_in_dataset


def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
    return True


def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)
        yield data


def sample_episodes(episodes, length, seed=0):
    random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        p = p / np.sum(p)
        while size < length:
            episode = random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(random.randint(0, total - 1))
                ret = {
                    k: v[index : min(index + length, total)] for k, v in episode.items()
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                ret = {
                    k: np.append(
                        ret[k], v[index : min(index + possible, total)], axis=0
                    )
                    for k, v in episode.items()
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret


def load_episodes(directory, limit=None, reverse=True):
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        ## 为了统一格式，我把好几个地方的keepdim都给去了
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=False))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=False))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=False)
        return (target * log_pred).sum(-1)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return (_mode.detach() + self._dist.mean - self._dist.mean.detach()).squeeze(-1)

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):