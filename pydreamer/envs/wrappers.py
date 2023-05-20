import time
from logging import exception
from typing import Callable

import gym
import gym.spaces
import numpy as np
import datetime
import uuid


class DictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = ...  # TODO

    def observation(self, obs):
        if isinstance(obs, dict):
            return obs  # Already a dictionary
        if len(obs.shape) == 1:
            return {'vecobs': obs}  # Vector env
        else:
            return {'image': obs}  # Image env


class TimeLimitWrapper(gym.Wrapper):

    def __init__(self, env, time_limit):
        super().__init__(env)
        self.time_limit = time_limit

    def step(self, action):
        obs, reward, done, info = self.env.step(action)  # type: ignore
        self.step_ += 1
        # print(self.step_)
        if self.step_ >= self.time_limit:
            done = True
            info['time_limit'] = True
        return obs, reward, done, info

    def reset(self):
        self.step_ = 0
        return self.env.reset()  # type: ignore


class ActionRewardResetWrapper(gym.Wrapper):

    def __init__(self, env, no_terminal):
        super().__init__(env)
        self.env = env
        self.no_terminal = no_terminal
        # Handle environments with one-hot or discrete action, but collect always as one-hot
        