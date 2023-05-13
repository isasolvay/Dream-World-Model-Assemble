import time
from logging import exception
from typing import Callable

import gym
import gym.spaces
import numpy as np
import datetime
import uuid


class DictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__