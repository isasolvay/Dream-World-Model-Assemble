import threading

import gym
import gym.envs.atari
import gym.wrappers
import numpy as np


class Atari_v2(gym.Env):

    LOCK = threading.Lock()

    def __init__(self,
                 name,
                 action_repeat=4,
                 size=(64, 64),
                 grayscale=False,  # DreamerV2 uses grayscale=Tr