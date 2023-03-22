from typing import Tuple

import gym
import gym.spaces
import gym_minigrid
import gym_minigrid.envs
import gym_minigrid.minigrid
from gym_minigrid.minigrid import COLOR_TO_IDX, OBJECT_TO_IDX
import numpy as np


class MiniGrid(gym.Env):

    GRID_VALUES = np.array([  # shape=(33,3)
        [0, 0, 0],  # Invisible
        [1, 0, 0],  # Empty
        [2, 5, 0],  # Wall
        [8, 1, 0],  # Goal
        # Agent
        [10, 0, 0],
        [10, 0, 1],
        [10, 0, 2],
        [10, 0, 3],
        # Door (color, state)
        [4, 0, 0],
        [4, 0, 1],
        [4, 1, 0],
        [4, 1, 1],
        [4, 2, 0],
        [4, 2, 1],
        [4, 3, 0],
        [4, 3, 1],
        [4, 4, 0],
        [4, 4, 1],
        [4, 5, 0],
        [4, 5, 1],
        # Key (color)
        [5, 0, 0],
        [5, 1, 0],
        [5, 2, 0],
        [5, 3, 0],
        [5, 4, 0],
        [5, 5, 0],
        # Ball (color)
        [6, 0, 0],
        [6, 1, 0],
        [6, 2, 0],
        [6, 3, 0],
        [6, 4, 0],
        [6, 5, 0],
        # Box (color)
        [7, 0, 0],
        [7, 1, 0],
        [7, 2, 0],
        [7, 3, 0],
        [7, 4, 0],
        [7, 5, 0],
    ])

    def __init__(self, env_name, max_steps=500, seed=None, agent_init_pos=None, agent_init_dir=0):
        env = gym.make(env_name)
        assert isinstance(env, gym_minigrid.envs.MiniGridEnv)
        self.env = env
        self.env.max_steps = max_steps
        if seed:
            self.env.seed(seed)
        self.max_steps = max_steps
        self.agent_init_pos = agent_init_pos
        self.agent_init_dir = agent_init_dir

        grid = self.env.grid.encode()  # type: ignore  # Grid is already generated when env is created
        self.map_size = n = grid.shape[0]
        self.map_centered_size = m = 2 * n - 3  # 11x11 => 19x19

        spaces = {}
        spaces['image'] = gym.spaces.Box(0, 255, (7, 7), np.uint8)
        spaces['map'] = gym.spaces.Box(0, 255, (n, n), np.uint8)
        spaces['map_agent'] = gym.spaces.Box(0, 255, (n, n), np.uint8)
        spaces['map_masked'] = gym.spaces.Box(0, 2