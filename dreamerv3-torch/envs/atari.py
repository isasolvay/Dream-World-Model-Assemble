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
        self._size = size
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
        img_shape = self._size + ((1,) if sel