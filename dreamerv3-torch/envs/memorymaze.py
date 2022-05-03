import gym
import numpy as np

###from tf dreamerv2 code


class MemoryMaze:
    def __init__(self, task, obs_key="image", act_key="action", size=(64, 64)):
        if task == "9x9":
            self._env = gym.make("memory_maze:MemoryMaze-9x9-v0")
 