import threading

import gym
import gym.envs.atari
import gym.wrappers
import numpy as np


class Atari_v2(gym.Env):

    LOCK = threading.Lock()

    def __init__(self,
 