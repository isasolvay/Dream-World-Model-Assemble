import gym
import numpy as np


class Atari:
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=4,
        size=(84, 84),
        gray=True,
        noops=0,
        lives="unused",
        sticky=True,
        actions="all",
  