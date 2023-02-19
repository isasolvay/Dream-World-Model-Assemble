import functools
from typing import List

import gym
import gym.spaces
from pydreamer.tools import print_once

import embodied
from embodied.envs import load_single_env


class EmbodiedEnv(gym.Env):
    """gym.Env wrapper around embo