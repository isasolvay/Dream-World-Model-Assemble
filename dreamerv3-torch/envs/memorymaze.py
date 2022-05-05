import gym
import numpy as np

###from tf dreamerv2 code


class MemoryMaze:
    def __init__(self, task, obs_key="image", act_key="action", size=(64, 64)):
        if task == "9x9":
            self._env = gym.make("memory_maze:MemoryMaze-9x9-v0")
        elif task == "15x15":
            self._env = gym.make("memory_maze:MemoryMaze-15x15-v0")
        else:
            raise NotImplementedError(task)
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._size = size
        self._gray = False

    def __getattr__(self, name):
        if na