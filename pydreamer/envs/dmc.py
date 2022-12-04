import os

import gym
import gym.spaces
import numpy as np


class DMC_v2(gym.Env):

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        # os.environ['MUJOCO_GL'] = 'egl'
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only d