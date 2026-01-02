from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class RoutingConfig:
    n_nodes: int = 64
    grid_w: int = 8
    grid_h: int = 8
    src: int = 0
    dst: int = 63
    lmin_ms: float = 0.0
    lmax_ms: float = 5.0
    prop_ms: float = 1.0
    max_steps: int = 256
    seed: int = 0

def _grid_neighbors(i: int, w: int, h: int) -> List[int]:
    r, c = divmod(i, w)
    nbrs = []
    if r > 0: nbrs.append((r-1)*w + c)
    if r < h-1: nbrs.append((r+1)*w + c)
    if c > 0: nbrs.append(r*w + (c-1))
    if c < w-1: nbrs.append(r*w + (c+1))
    return nbrs

class SatelliteNetworkEnv(gym.Env):
    """Toy SatCom routing environment for tabular Q-routing.

    - State: current node id (satellite)
    - Action: next-hop node id (must be adjacent)
    - Reward: negative of propagation + sampled queueing latency, with terminal bonus when reaching dst.
    """
    metadata = {"render_modes": []}

    def __init__(self, config: RoutingConfig):
        super().__init__()
        self.cfg = config
        assert self.cfg.grid_w * self.cfg.grid_h == self.cfg.n_nodes, "grid size mismatch"
        self.rng = np.random.default_rng(self.cfg.seed)

        # adjacency list
        self.adj: Dict[int, List[int]] = {i: _grid_neighbors(i, self.cfg.grid_w, self.cfg.grid_h) for i in range(self.cfg.n_nodes)}

        self.observation_space = spaces.Discrete(self.cfg.n_nodes)
        self.action_space = spaces.Discrete(self.cfg.n_nodes)  # we will validate feasibility manually

        self.state: int = self.cfg.src
        self.steps = 0

        # per-link queue delay bounds (ms), randomly assigned but deterministic given seed
        self.link_bounds: Dict[Tuple[int,int], Tuple[float,float]] = {}
        for u in range(self.cfg.n_nodes):
            for v in self.adj[u]:
                lo = float(self.rng.uniform(self.cfg.lmin_ms, self.cfg.lmax_ms))
                hi = float(self.rng.uniform(lo, self.cfg.lmax_ms + 5.0))
                self.link_bounds[(u,v)] = (lo, hi)

    def reset(self, *, seed: Optional[int]=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.cfg.src
        self.steps = 0
        return self.state, {}

    def get_valid_actions(self, state: int) -> List[int]:
        return list(self.adj[state])

    def _sample_queue_ms(self, u: int, v: int) -> float:
        lo, hi = self.link_bounds[(u,v)]
        return float(self.rng.uniform(lo, hi))

    def step(self, action: int):
        self.steps += 1
        done = False
        trunc = False

        if action not in self.adj[self.state]:
            # invalid transition: strong penalty, keep state
            reward = -100.0
            if self.steps >= self.cfg.max_steps:
                trunc = True
            return self.state, reward, done, trunc, {"invalid": True}

        # latency model
        l_prop = self.cfg.prop_ms
        l_queue = self._sample_queue_ms(self.state, action)
        reward = -(l_prop + l_queue)

        # terminal shaping (bonus when reaching destination)
        if action == self.cfg.dst:
            reward += 100.0
            done = True

        self.state = action

        if self.steps >= self.cfg.max_steps and not done:
            trunc = True

        return self.state, reward, done, trunc, {"l_prop": l_prop, "l_queue": l_queue}
