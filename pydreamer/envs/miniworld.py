
import time
from logging import warning
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit

WALL = 2


class MazeBouncingBallPolicy:
    # Policy:
    #   1) Forward until you hit a wall
    #   2) Turn in random 360 direction
    #   3) Go to 1)

    def __init__(self):
        self.pos = None
        self.turns_remaining = 0

    def __call__(self, obs) -> Tuple[int, dict]:
        assert 'agent_pos' in obs, f'Need agent position'
        pos = obs['agent_pos']
        action = -1

        # print(f'{self.pos} => {pos} ({obs["agent_dir"]})')

        if self.turns_remaining == 0:
            if self.pos is None or not np.all(self.pos == pos):
                # Going forward
                action = 2
                self.pos = pos
            else:
                # Hit the wall - start turning
                if np.random.randint(2) == 0:
                    # self.turns_remaining = -np.random.randint(2, 5)  # Left
                    self.turns_remaining = -1  # TODO
                else:
                    # self.turns_remaining = np.random.randint(2, 5)  # Right
                    self.turns_remaining = 1  # TODO
                self.pos = None

        if self.turns_remaining > 0:
            # Turning right
            action = 1
            self.turns_remaining -= 1

        elif self.turns_remaining < 0:
            # Turning left
            action = 0
            self.turns_remaining += 1

        assert action >= 0
        return action, {}


class MazeDijkstraPolicy:
    # Policy:
    #   1) Pick a random spot on a map
    #   2) Go there using shortest path
    #   3) Occasionally perform a random action

    def __init__(self, step_size, turn_size, random_prob=0.02, random_steps=5, goal_strategy='random'):
        self.step_size = step_size
        self.turn_size = turn_size
        self.random_prob = random_prob
        self.random_steps = random_steps
        self.goal_strategy = goal_strategy
        self.goal = None
        self.expected_pos = None
        self.random_remaining = 0

    def __call__(self, obs) -> Tuple[int, dict]:
        assert 'agent_pos' in obs, 'Need agent position'
        assert 'map_agent' in obs, 'Need map'

        x, y = obs['agent_pos']
        dx, dy = obs['agent_dir']
        d = np.arctan2(dy, dx) / np.pi * 180
        map = obs['map']

        if obs['reset']:  # new episode
            self.goal = None
            self.expected_pos = None
            self.random_remaining = 0

        if self.goal is None:
            self.goal = self.generate_goal(obs)

        if self.expected_pos is not None:
            if not np.isclose(self.expected_pos[:2], np.array([x, y]), 1e-3).all():
                warning('Unexpected position - stuck? Performing random dance...')
                self.random_remaining = self.random_steps

        while True:
            t = time.time()
            actions, path, nvis = find_shortest(map, (x, y, d), self.goal, self.step_size, self.turn_size)
            # print(f'Pos: {tuple(np.round([x,y,d], 2))}'
            #       f', Goal: {self.goal}'
            #       f', Len: {len(actions)}'
            #       f', Actions: {actions[:1]}'
            #       # f', Path: {path[:1]}'
            #       f', Visited: {nvis}'
            #       f', Time: {int((time.time()-t)*1000)}'
            #       )
            if actions is None:
                warning(f'No path found from=({x:.2f}, {y:.2f}, {d:.2f})  to={self.goal}  nvis={nvis} - trying new goal...')
                self.goal = self.generate_goal(obs)

            elif len(actions) == 0:
                # Goal reached
                self.goal = self.generate_goal(obs)

            else:
                assert path is not None
                if np.random.rand() < self.random_prob:  # initialize random action sequence
                    self.random_remaining = self.random_steps
                if self.random_remaining > 0:  # continue random action
                    self.random_remaining -= 1
                    self.expected_pos = None
                    return np.random.randint(3), {}
                else:
                    self.expected_pos = path[0]
                    return actions[0], {}  # shortest-path action

    def generate_goal(self, obs) -> Tuple[float, float]:
        map = obs['map']
        x, y = obs['agent_pos']
        dx, dy = obs['agent_dir']
        d = np.arctan2(dy, dx)

        if self.goal_strategy == 'random':
            while True:
                x = np.random.randint(map.shape[0])
                y = np.random.randint(map.shape[1])
                if map[x, y] != WALL:
                    return x, y

        if self.goal_strategy == 'goal_direction':
            grx, gry = obs['goal_direction']  # agent-relative
            gx = x + grx * np.cos(d) - gry * np.sin(d)  # convert to absolute
            gy = y + gry * np.cos(d) + grx * np.sin(d)
            return (gx, gy)

        assert False, self.goal_strategy


@njit
def find_shortest(map, start, goal, step_size=1.0, turn_size=45.0):
    KPREC = 5
    RADIUS = 0.2
    x, y, d = start
    gx, gy = goal

    # Well ok, this is BFS not Dijkstra, technically speaking

    que = []
    que_ix = 0
    visited = {}
    parent = {}
    parent_action = {}

    p = (x, y, d)
    key = (round(x * KPREC) / KPREC, round(y * KPREC) / KPREC, round(d * KPREC) / KPREC)
    que.append(p)
    visited[key] = True
    goal_state = None

    while que_ix < len(que):
        p = que[que_ix]
        que_ix += 1
        x, y, d = p
        if np.sqrt((x - gx) ** 2 + (y - gy) ** 2) < step_size:
            goal_state = p
            break
        for action in range(3):
            x1, y1, d1 = x, y, d
            if action == 0:  # turn left
                d1 = d - turn_size
                if d1 < -180.0:
                    d1 += 360.0
            if action == 1:  # turn right
                d1 = d + turn_size
                if d1 > 180.0:
                    d1 -= 360.0
            if action == 2:  # forward
                x1 = x + step_size * np.cos(d / 180 * np.pi)
                y1 = y + step_size * np.sin(d / 180 * np.pi)
                # Check wall collision at 4 corners
                for x2, y2 in [(x1 - RADIUS, y1 - RADIUS), (x1 + RADIUS, y1 - RADIUS), (x1 - RADIUS, y1 + RADIUS), (x1 + RADIUS, y1 + RADIUS)]:
                    if x2 < 0 or y2 < 0 or x2 >= map.shape[0] or y2 >= map.shape[1] or map[int(x2), int(y2)] == WALL:
                        x1, y1 = x, y  # wall
                        break
            p1 = (x1, y1, d1)
            key = (round(x1 * KPREC) / KPREC, round(y1 * KPREC) / KPREC, round(d1 * KPREC) / KPREC)
            if key not in visited:
                que.append(p1)
                parent[p1] = p
                parent_action[p1] = action
                visited[key] = True
                assert len(visited) < 100000, 'Runaway Dijkstra?'

    if goal_state is None:
        return None, None, len(visited)

    path = []
    actions = []
    p = goal_state
    while p in parent_action:
        path.append(p)
        actions.append(parent_action[p])
        p = parent[p]
    path.reverse()
    actions.reverse()
    return actions, path, len(visited)