import functools
from typing import List

import gym
import gym.spaces
from pydreamer.tools import print_once

import embodied
from embodied.envs import load_single_env


class EmbodiedEnv(gym.Env):
    """gym.Env wrapper around embodied.Env"""

    def __init__(self,
                 task,
                 action_repeat=1,
                 time_limit=0,
                 obs_keys=['image', 'inventory', 'equipped'],  # TODO: this default is for Minecraft
                 restart=True,  # restart needed for Min