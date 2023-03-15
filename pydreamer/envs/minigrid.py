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
        [4, 