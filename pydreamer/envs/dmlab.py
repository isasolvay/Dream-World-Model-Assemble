# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepMind Lab Gym wrapper."""

import os
import gym
import gym.spaces
import numpy as np
from PIL import Image

import deepmind_lab  # type: ignore

# Default (action_dim=9)
# ACTION_SET = (  # IMPALA action set
#     (0, 0, 0, 1, 0, 0, 0),    # Forward
#     (0, 0, 0, -1, 0, 0, 0),   # Backward
#     (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
#     (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
#     (-20, 0, 0, 0, 0, 0, 0),  # Look Left
#     (20, 0, 0, 0, 0, 0, 0),   # Look Right
#     (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
#     (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
#     (0, 0, 0, 0, 1, 0, 0),    # Fire.
# )

# RLU (action:dim=15)
ACTION_SET = {  # R2D2 action set
    0: (0, 0, 0, 1, 0, 0, 0),     # Forward
    1: (0, 0, 0, -1, 0, 0, 0),    # Backward
    2: (0, 0, -1, 0, 0, 0, 0),    # Strafe Left
    3: (0, 0, 1, 0, 0, 0, 0),     # Strafe Right
    4: (-10, 0, 0, 0, 0, 0, 0),   # Left (10 deg)
    5: (10, 0, 0, 0, 0, 0, 0),    # Right (10 deg)
    6: (-60, 0, 0, 0, 0, 0, 0),   # Left (60 deg)
    7: (60, 0, 0, 0, 0, 0, 0),    # Right (60 deg)
    8: (0, 10, 0, 0, 0, 0, 0),    # Up (10 deg)
    9: (0, -10, 0, 0, 0, 0, 0),   # Down (10 deg)
    10: (-10, 0, 0, 1, 0, 0, 0),  # Left (10 deg) + Forward
    11: (10, 0, 0, 1, 0, 0, 0),   # Right (10 deg) + Forward
    12: (-60, 0, 0, 1, 0, 0, 0),  # Left (60 deg) + Forward
    13: (60, 0, 0, 1, 0, 0, 0),   # Right (60 deg) + Forward
    14: (0, 0, 0, 0, 1, 0, 0),    # Fire
}

ALL_GAMES = frozenset([
    'rooms_collect_good_objects_train',  # rooms_collect_good_objects
    'rooms_collect_good_objects_test',  # rooms_collect_good_objects
    'rooms_exploit_deferred_effects_train',  # rooms_exploit_deferred_effects
    'rooms_exploit_deferred_effects_test',  # rooms_exploit_deferred_effects
    'rooms_select_nonmatching_object',
    'rooms_watermaze',
    'rooms_keys_doors_puzzle',
    'language_select_described_object',
    'language_select_located_object',
    'language_execute_random_task',
    'language_answer_quantitative_question',
    'lasertag_one_opponent_small',
    'lasertag_three_opponents_small',
    'lasertag_one_opponent_large',
    'lasertag_three_opponents_large',
    'natlab_fixed_large_map',
    'natlab_varying_map_regrowth',
    'natlab_varying_map_randomized',
    'skymaze_irreversible_path_hard',
    'skymaze_irreversible_path_varied',
    'psychlab_arbitrary_visuomotor_mapping',
    'psychlab_continuous_recognition',
    'psychlab_sequential_comparison',
    'psychlab_visual_search',
    'explore_object_locations_small',
    'explore_object_locations_large',
    'explore_obstruc