
import numpy as np
from . import minecraft_base

import gym


def make_env(task, *args, **kwargs):
    return {
        "wood": MinecraftWood,
        "climb": MinecraftClimb,
        "diamond": MinecraftDiamond,
    }[task](*args, **kwargs)


class MinecraftWood:
    def __init__(self, *args, **kwargs):
        actions = BASIC_ACTIONS
        self.rewards = [
            CollectReward("log", repeated=1),
            HealthReward(),
        ]
        env = minecraft_base.MinecraftBase(actions, *args, **kwargs)

    def step(self, action):