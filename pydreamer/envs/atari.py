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
                 grayscale=False,  # DreamerV2 uses grayscale=True
                 noops=30,
                 life_done=False,
                 sticky_actions=True,
                 all_actions=True
                 ):
        assert size[0] == size[1]
        with self.LOCK:
            env = gym.envs.atari.AtariEnv(
               