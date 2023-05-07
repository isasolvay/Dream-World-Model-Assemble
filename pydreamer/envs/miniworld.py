
import time
from logging import warning
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit

WALL = 2


class MazeBouncingBallPolicy:
    # Policy:
    #   1) Forward until you hit a wall