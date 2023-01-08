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
    5: (10, 0, 0, 0, 0, 0, 0),    # Right (10 deg