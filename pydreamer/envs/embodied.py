import functools
from typing import List

import gym
import gym.spaces
from pydreamer.tools import print_once

import embodied
from embodied.envs import load_single_env


class EmbodiedEnv(gym.Env):
    """gym.Env wrapper around embodied.Env"""

    def __init__(self,
                 task,
                 action_repeat=1,
                 time_limit=0,
                 obs_keys=['image', 'inventory', 'equipped'],  # TODO: this default is for Minecraft
                 restart=True,  # restart needed for Minecraft
                 ):
        ctor = functools.partial(load_single_env, task, repeat=action_repeat, length=time_limit)
        if restart:
            self._env = embodied.wrappers.RestartOnException(ctor)
        else:
            self._env = ctor()
        self.obs_keys = obs_keys
        self.action_space = EmbodiedEnv.convert_act_space(self._env.act_space)
        self.observation_space = EmbodiedEnv.convert_obs_space(self._env.obs_space, obs_keys)

    @staticmethod
    def convert_act_space(act_space: embodied.Space) -> gym.spaces.Space:
        assert isinstance(act_space, dict)
        assert tuple(act_space.keys()) == ('action', 'reset')
        return space_from_embodied(act_space['action'])

    @staticmethod
    def convert_obs_space(obs_space: embodied.Space, obs_keys: List[str]) -> gym.spaces.Space:
        assert isinstance(obs_space, dict)
        print_once(f'Using observation keys {obs_keys} from available:', list(obs_space.keys()))
        assert 'reward' in obs_space
        assert 'is_first' in obs_space
        assert 'is_last' in obs_space
        assert 'is_terminal' in obs_space
        gym_space = gym.spaces.Dict({k: space_from_embodied(obs_space[k]) for k in obs_keys})
        return gym_space

    def reset(self):
        ts = self._env.step({'action': self.action_space.low, 'reset': True})  # type: ignore
        obs, reward, done, info = self._obs(ts)
        return obs

    def step(self, action):
        ts = self._env.step({'action': action, 'reset': False})
     