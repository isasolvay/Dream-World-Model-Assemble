
# Copyright danijar, jurgisp

import gym
import gym.spaces
import minerl
import numpy as np


def make_action(pitch=0, yaw=0, **kwargs):
    action = dict(
        camera=[pitch, yaw],
        forward=0, back=0, left=0, right=0,
        attack=0, sprint=0, jump=0, sneak=0)
    action.update(kwargs)
    return action


BASIC_ACTIONS = (
    make_action(),
    make_action(pitch=-10),
    make_action(pitch=10),
    make_action(yaw=-30),
    make_action(yaw=30),
    make_action(attack=1),
    make_action(forward=1),
    make_action(back=1),
    make_action(left=1),
    make_action(right=1),
    make_action(sprint=1),