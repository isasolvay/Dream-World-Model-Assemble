
import logging
import threading

import numpy as np
import gym


class MinecraftBase(gym.Env):
    _LOCK = threading.Lock()

    def __init__(
        self,
        actions,
        repeat=1,
        size=(64, 64),
        break_speed=100.0,
        gamma=10.0,
        sticky_attack=30,
        sticky_jump=10,
        pitch_limit=(-60, 60),
        logs=True,
    ):
        if logs:
            logging.basicConfig(level=logging.DEBUG)
        self._repeat = repeat
        self._size = size
        if break_speed != 1.0:
            sticky_attack = 0

        # Make env
        with self._LOCK:
            from . import minecraft_minerl

            self._env = minecraft_minerl.MineRLEnv(size, break_speed, gamma).make()