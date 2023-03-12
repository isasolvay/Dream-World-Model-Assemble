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
