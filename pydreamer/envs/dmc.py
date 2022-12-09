import os

import gym
import gym.spaces
import numpy as np


class DMC_v2(gym.Env):

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        # os.environ['MUJOCO_GL'] = 'egl'
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if domain == 'manip':
            from dm_control import manipulation
            self._env = manipulation.load(task + '_vision')
        elif domain == 'locom':
            from dm_control.locomotion.examples import basic_rodent_2020
            self._env = getattr(basic_rodent_2020, task)()
        else:
            from dm_control import suite
            self._env = suite.load(domain, task)
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(
                quadruped_walk=2,
                quadruped_run=2,
                quadruped_escape=2,
                quadruped_fetch=2,
                locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera
        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
            