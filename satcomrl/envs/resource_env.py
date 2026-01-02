from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

ACTIONS = ["NOP", "Q", "D", "QD"]  # 0..3

@dataclass
class ResourceConfig:
    num_sats: int = 4
    num_slots: int = 40
    seed: int = 0

    # Rates per second (scaled by slot duration tau)
    dB_bg: float = -0.00002
    dB_dl: float = -0.00025
    dB_sun: float = 0.00030

    dM_tm: float = 0.00001
    dM_dl: float = -0.00040
    dM_aq: float = 0.00035

    # Reward rates per second
    r_aq: float = 1.0
    r_dl: float = 1.0

    # Penalties
    p_bat: float = 50.0
    p_mem: float = 50.0
    p_inv: float = 5.0
    p_sim_gs: float = 2.0
    p_sim_at: float = 2.0

    # If True, opportunity schedule is random but reproducible; else deterministic simple pattern.
    random_opps: bool = True

def encode_action_tuple(action_tuple: Tuple[int, ...]) -> int:
    # base-4 encoding
    a = 0
    for x in action_tuple:
        a = a * 4 + int(x)
    return a

def decode_action_id(action_id: int, n: int) -> Tuple[int, ...]:
    out = [0]*n
    x = int(action_id)
    for i in range(n-1, -1, -1):
        out[i] = x % 4
        x //= 4
    return tuple(out)

class SatelliteResourceEnv(gym.Env):
    """Toy multi-satellite resource allocation env with action masking.

    Observation is a dict:
      - obs: flattened vector [tau, per-sat(batt, mem, opp_at, opp_gs, sun)]  (opp_* are integers)
      - action_mask: binary vector of length 4^N over feasible actions wrt opp availability (not battery/mem)

    Action is Discrete(4^N): joint action for all satellites.
    """
    metadata = {"render_modes": []}

    def __init__(self, config: ResourceConfig):
        super().__init__()
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.seed)

        self.action_dim = 4 ** self.cfg.num_sats
        self.action_space = spaces.Discrete(self.action_dim)

        # Build observation space
        # tau (0..1 scaled), batt/mem in [0,1], opp_at/opp_gs in [0..K], sun in {0,1}
        # We'll store opps as floats for NN input (normalized).
        obs_len = 1 + self.cfg.num_sats * 5
        self._obs_len = obs_len
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32),
            "action_mask": spaces.Box(low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        })

        self.t = 0
        self.tau = 60.0  # seconds
        self.batt = np.ones(self.cfg.num_sats, dtype=np.float32) * 0.8
        self.mem = np.ones(self.cfg.num_sats, dtype=np.float32) * 0.2

        # opportunities per slot per sat: (opp_at_id, opp_gs_id, sun)
        self.opp_at = np.zeros((self.cfg.num_slots, self.cfg.num_sats), dtype=np.int32)
        self.opp_gs = np.zeros((self.cfg.num_slots, self.cfg.num_sats), dtype=np.int32)
        self.sun = np.zeros((self.cfg.num_slots, self.cfg.num_sats), dtype=np.int32)

        self._generate_opportunities()

    def _generate_opportunities(self):
        K_at = max(3, self.cfg.num_sats)  # toy ids
        K_gs = max(2, self.cfg.num_sats//2)

        for t in range(self.cfg.num_slots):
            for i in range(self.cfg.num_sats):
                if self.cfg.random_opps:
                    self.sun[t, i] = int(self.rng.random() < 0.55)
                    self.opp_at[t, i] = int(self.rng.integers(0, K_at+1)) if self.rng.random() < 0.35 else 0
                    self.opp_gs[t, i] = int(self.rng.integers(0, K_gs+1)) if self.rng.random() < 0.30 else 0
                else:
                    self.sun[t, i] = 1 if (t + i) % 3 != 0 else 0
                    self.opp_at[t, i] = (i % K_at) + 1 if (t % 5 == 0) else 0
                    self.opp_gs[t, i] = (i % K_gs) + 1 if (t % 7 == 0) else 0

    def reset(self, *, seed: Optional[int]=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.tau = 60.0
        self.batt[:] = 0.8
        self.mem[:] = 0.2
        return self._get_obs(), {}

    def _feasible_single(self, a: int, has_at: bool, has_gs: bool) -> bool:
        if a == 0:  # NOP
            return True
        if a == 1:  # Q
            return has_at
        if a == 2:  # D
            return has_gs
        if a == 3:  # QD
            return has_at and has_gs
        return False

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros((self.action_dim,), dtype=np.float32)
        has_at = self.opp_at[self.t] != 0
        has_gs = self.opp_gs[self.t] != 0
        for aid in range(self.action_dim):
            tup = decode_action_id(aid, self.cfg.num_sats)
            ok = True
            for i, a in enumerate(tup):
                if not self._feasible_single(a, bool(has_at[i]), bool(has_gs[i])):
                    ok = False
                    break
            mask[aid] = 1.0 if ok else 0.0
        return mask

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # normalize tau into [0,1] using 120s cap
        tau_n = np.clip(self.tau / 120.0, 0.0, 1.0)
        vec = [tau_n]
        for i in range(self.cfg.num_sats):
            vec.extend([
                float(self.batt[i]),
                float(self.mem[i]),
                float(self.opp_at[self.t, i] > 0),
                float(self.opp_gs[self.t, i] > 0),
                float(self.sun[self.t, i]),
            ])
        obs = np.array(vec, dtype=np.float32)
        return {"obs": obs, "action_mask": self._action_mask()}

    def step(self, action: int):
        assert self.action_space.contains(action)
        tup = decode_action_id(action, self.cfg.num_sats)
        mask = self._action_mask()
        invalid = (mask[action] < 0.5)

        # slot duration: vary slightly
        self.tau = float(self.rng.uniform(30.0, 90.0))
        tau = self.tau

        # contention penalties: if multiple sats choose same AT/GS simultaneously
        chosen_at = []
        chosen_gs = []
        for i, a in enumerate(tup):
            if a in (1,3) and self.opp_at[self.t, i] != 0:
                chosen_at.append(self.opp_at[self.t, i])
            if a in (2,3) and self.opp_gs[self.t, i] != 0:
                chosen_gs.append(self.opp_gs[self.t, i])

        # count duplicates
        sim_at_pen = 0.0
        sim_gs_pen = 0.0
        if len(chosen_at) > 1:
            sim_at_pen = self.cfg.p_sim_at * (len(chosen_at) - len(set(chosen_at)))
        if len(chosen_gs) > 1:
            sim_gs_pen = self.cfg.p_sim_gs * (len(chosen_gs) - len(set(chosen_gs)))

        # Dynamics + reward
        reward = 0.0
        for i, a in enumerate(tup):
            has_at = self.opp_at[self.t, i] != 0
            has_gs = self.opp_gs[self.t, i] != 0
            sun = self.sun[self.t, i]

            chi_aq = 1.0 if (a in (1,3) and has_at) else 0.0
            chi_dl = 1.0 if (a in (2,3) and has_gs) else 0.0

            # update battery/memory
            batt_next = self.batt[i] + tau*(self.cfg.dB_bg + chi_dl*self.cfg.dB_dl + sun*self.cfg.dB_sun)
            mem_next  = self.mem[i]  + tau*(self.cfg.dM_tm + chi_dl*self.cfg.dM_dl + chi_aq*self.cfg.dM_aq)

            # reward rates
            reward += tau*(chi_aq*self.cfg.r_aq + chi_dl*self.cfg.r_dl)

            # penalties for violations
            bat_pen = 0.0
            mem_pen = 0.0
            if batt_next < 0.0:
                bat_pen = self.cfg.p_bat * abs(batt_next)
            if mem_next > 1.0:
                mem_pen = self.cfg.p_mem * abs(mem_next - 1.0)

            reward -= (bat_pen + mem_pen)

            # clip states
            self.batt[i] = float(np.clip(batt_next, 0.0, 1.0))
            self.mem[i]  = float(np.clip(mem_next, 0.0, 1.0))

        if invalid:
            reward -= self.cfg.p_inv

        reward -= (sim_at_pen + sim_gs_pen)

        self.t += 1
        done = self.t >= self.cfg.num_slots
        trunc = False

        info = {"invalid": invalid, "tau": tau, "sim_at_pen": sim_at_pen, "sim_gs_pen": sim_gs_pen}
        obs = self._get_obs() if not done else {"obs": np.zeros((self._obs_len,), dtype=np.float32),
                                                "action_mask": np.ones((self.action_dim,), dtype=np.float32)}
        return obs, float(reward), done, trunc, info
