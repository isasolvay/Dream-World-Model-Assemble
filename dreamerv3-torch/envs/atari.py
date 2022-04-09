import gym
import numpy as np


class Atari:
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name,
        action_repea