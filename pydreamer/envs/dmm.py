
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import json
import os
import random
import time
from logging import debug, info, warning
from typing import NamedTuple

import dm_env
import grpc
import gym
import gym.spaces
import numpy as np
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor, dm_env_rpc_pb2
from dm_env_rpc.v1 import error as dm_env_rpc_error
from dm_env_rpc.v1 import tensor_utils
from PIL import Image

ACTION_SET = [
    {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': +1, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': -1, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': +1, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': -1, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': +1, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': -1, 'LOOK_DOWN_UP': 0},
    # {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': +1},
    # {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': -1},
    {'MOVE_BACK_FORWARD': +1, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': +1, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': +1, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': -1, 'LOOK_DOWN_UP': 0},
]

_MEMORY_TASK_LEVEL_NAMES = [
    'spot_diff_motion_train',
    'spot_diff_multi_train',
    'spot_diff_passive_train',
    'spot_diff_train',
    'invisible_goal_empty_arena_train',