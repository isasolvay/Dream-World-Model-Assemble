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
  