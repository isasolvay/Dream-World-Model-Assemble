import threading

import gym
import gym.envs.atari
import gym.wrappers
import numpy as np


class Atari_v2(gym.Env):

    LOCK = threading.Lock()

    def __init__(self,
                 name,
                 action_repeat=4,
                 size=(64, 64),
                 grayscale=False,  # DreamerV2 uses grayscale=True
                 noops=30,
                 life_done=False,
                 sticky_actions=True,
                 all_actions=True
                 ):
        assert size[0] == size[1]
        with self.LOCK:
            env = gym.envs.atari.AtariEnv(
                game=name,
                obs_type='image',
                frameskip=1,
                repeat_action_probability=0.25 if sticky_actions else 0.0,
                full_action_space=all_actions)
        # Avoid unnecessary rendering in inner env.
        env.get_obs = lambda: None  # type: ignore
        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')  # type: ignore
        env = gym.wrappers.AtariPreprocessing(env, noops, action_repeat, size[0], life_done, grayscale)
        self.env = env
        self.grayscale = grayscale

    @property
    def observation_space(self):
        return gym.spaces.Dict({'image': self.env.observation_space})  # type: ignore

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        with self.LOCK:
            image: np.ndarray = self.env.reset()  # type: ignore
        if self.grayscale:
            image = image[..., None]
        obs = {'image': image}
        return obs

    def step(self, action):
        image, reward, done, info = self.env.step(action)
        if self.grayscale:
            image = image[..., None]
        obs = {'image': image}
        return obs, reward, done, info

    def render(self, mode):
        return self.env.render(mode)
    
class Atari_v3:
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
        length=108000,
        resize="opencv",
        seed=None,
    ):
        assert size[0] == size[1]
        assert lives in ("unused", "discount", "reset"), lives
        assert actions in ("all", "needed"), actions
        assert resize in ("opencv", "pillow"), resize
        if self.LOCK is None:
            import multiprocessing as mp

            mp = mp.get_context("spawn")
            self.LOCK = mp.Lock()
        self._resize = resize
        if self._resize == "opencv":
            import cv2

            self._cv2 = cv2
        if self._resize == "pillow":
            from PIL import Image

            self._image = Image
        import gym.envs.atari

        if name == "james_bond":
            name = "jamesbond"
        self._repeat = action_repeat
        self._size = tuple(size)
        self._gray = gray
        self._noops = noops
        self._lives = lives
        self._sticky = sticky
        self._length = length
        self._random = np.random.RandomState(seed)
        with self.LOCK:
            self._env = gym.envs.atari.AtariEnv(
                game=name,
                obs_type="image",
                frameskip=1,
                repeat_action_probability=0.25 if sticky else 0.0,
                full_action_space=(actions == "all"),
            )
        assert self._env.unwrapped.get_action_meanings()[0] == "NOOP"
        shape = self._env.observation_space.shape
        self._buffer = [np.zeros(shape, np.uint8) for _ in range(2)]
        self._ale = self._env.unwrapped.ale
        self._last_lives = None
        self._done = True
        self._step = 0
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size + ((1,) if self._gray else (3,))
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
            }
        )

    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = True
        return space

    def step(self, action):
        # if action['reset'] or self._done:
        #   with self.LOCK:
        #     self._reset()
        #   self._done = False
        #   self._step = 0
        #   return self._obs(0.0, reset=True)
        total = 0.0
        dead = False
        if not isinstance(action, int):
            if len(action.shape) >= 1:
                action = np.argmax(action)
        for repeat in range(self._repeat):
            _, reward, over, info = self._env.step(action)
            self._step += 1
            total += reward
            if repeat == self._repeat - 2:
                self._screen(self._buffer[1])
            if over:
                break
            if self._lives != "unused":
                current = self._ale.lives()
                if current < self._last_lives:
                    dead = True
                    self._last_lives = current
                    break
        if not self._repeat:
            self._buffer[1][:] = self._buffer[0][:]
        self._screen(self._buffer[0])
        self._done = over or (self._length and self._step >= self._length)
        return self._obs(
            total,
            is_last=self._done or (dead and self._lives == "reset"),
            terminal=dead or over,
        )

    def reset(self):
        self._env.reset()
        if self._noops:
            for _ in range(self._random.randint(self._noops)):
                _, _, dead, _ = self._env.step(0)
                if dead:
                    self._env.reset()
        self._last_lives = self._ale.lives()
        self._screen(self._buffer[0])
        self._buffer[1].fill(0)

        self._done = False
        self._step = 0
        obs, reward, terminal, _ = self._obs(0.0, reset=True)
        return obs

    def _obs(self, reward, reset=False, is_last=False, terminal=False):
        np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0])
        image = self._buffer[0]
        if image.shape[:2] != self._size:
            if self._resize == "opencv":
   