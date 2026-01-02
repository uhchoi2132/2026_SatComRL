from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from satcomrl.envs.routing_env import SatelliteNetworkEnv, RoutingConfig

@dataclass
class QRoutingParams:
    episodes: int = 20000
    alpha: float = 0.05
    gamma: float = 0.9
    epsilon: float = 0.3
    epsilon_decay: float = 0.9999

def train_qrouting(env: SatelliteNetworkEnv, params: QRoutingParams) -> np.ndarray:
    n = env.cfg.n_nodes
    Q = np.zeros((n, n), dtype=np.float32)
    eps = params.epsilon

    for ep in range(params.episodes):
        state, _ = env.reset()
        visited = set()
        done = False
        while not done:
            visited.add(state)
            valid = env.get_valid_actions(state)
            if env.rng.random() < eps:
                action = int(env.rng.choice(valid))
            else:
                qvals = [Q[state, a] for a in valid]
                action = int(valid[int(np.argmax(qvals))])

            new_state, reward, done, trunc, info = env.step(action)

            # loop penalty (optional)
            if new_state in visited and not done:
                reward -= 100.0
                done = True

            # Q update
            next_valid = env.get_valid_actions(new_state) if not done else []
            max_next = np.max([Q[new_state, a] for a in next_valid]) if next_valid else 0.0
            Q[state, action] = (1 - params.alpha) * Q[state, action] + params.alpha * (reward + params.gamma * max_next)

            state = new_state
            if trunc:
                break

        eps *= params.epsilon_decay

    return Q

def greedy_route_from_Q(env: SatelliteNetworkEnv, Q: np.ndarray) -> List[int]:
    state, _ = env.reset()
    route = [state]
    visited = set([state])
    while state != env.cfg.dst and len(route) < env.cfg.max_steps:
        valid = env.get_valid_actions(state)
        qvals = [(a, Q[state, a]) for a in valid]
        qvals.sort(key=lambda x: x[1], reverse=True)
        nxt = None
        for a, _ in qvals:
            if a not in visited:
                nxt = a
                break
        if nxt is None:
            nxt = qvals[0][0]
        state, reward, done, trunc, info = env.step(nxt)
        route.append(state)
        visited.add(state)
        if done or trunc:
            break
    return route
