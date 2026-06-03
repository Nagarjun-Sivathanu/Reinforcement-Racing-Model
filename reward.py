"""Waypoint-following observation + checkpoint-gated progress reward for DonkeySim RL.

Design (agreed with the user): the policy must *see the road ahead* so that one trained brain
can drive a different track by swapping in that track's waypoints. The observation is built in
the car's own frame from a per-track centerline (see track_utils + record_track.py):

    [ signed_cte, speed, forward_vel, last_steer, last_throttle,
      (lateral, forward) of the next N lookahead waypoints ... ]

Everything is track-relative, so it transfers across tracks.

Reward = dense forward **progress** along the path + a small speed term + ordered **checkpoint
(sector) bonuses** + a checkpoint-gated **lap bonus** + a terminal penalty. There is no
centring term (the racing line is free; track limits are enforced by max_cte termination).

Anti-exploit: laps/sectors are detected from OUR OWN monotonic forward progress, NOT the sim's
start-line counter. The sim increments lap_count on *every* start-line crossing (even reversing
back and forth over it), which an agent will happily farm. Here, a checkpoint is a *progress
threshold* (a virtual line across the road, not a point), checkpoints must be crossed IN ORDER,
each pays at most once per lap, and the lap bonus only unlocks after a full loop of progress.
Wiggling at the start line advances no progress, so it earns nothing. The time bonus is capped.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from track_utils import ground_xz, nearest_index, signed_cte, lookahead_local

# --- reward weights (tune these) -------------------------------------------
K_PROGRESS = 1.0          # reward per world-unit advanced along the path (the workhorse)
K_SPEED = 0.05            # small bonus per unit forward velocity (kept low: see TIME_PENALTY)
TERMINAL_PENALTY = -10.0  # ending the episode (off-track / crash) -- kept strong on purpose:
                          # it's what makes braking worthwhile and prevents reckless full-send
TIME_PENALTY = -0.05      # small per-step cost: kills crawling/dawdling without full-send,
                          # because the strong terminal penalty still punishes crashing
K_SMOOTH = 0.1            # penalty on step-to-step action change (anti-jitter / anti-panic)
SECTOR_BONUS = 5.0        # once-per-lap, in-order bonus for crossing a checkpoint line
LAP_BONUS_FLAT = 50.0     # flat reward for a *real* (checkpoint-gated) completed lap
LAP_BONUS_TIME = 300.0    # time bonus = LAP_BONUS_TIME / lap_seconds, capped below
LAP_TIME_CAP = 50.0       # hard cap on the time bonus (prevents any blow-up exploit)

# --- checkpoints / observation shape ---------------------------------------
NUM_CHECKPOINTS = 8       # progress gates per lap ("lines across the road", not points)
SIM_DT = 0.05             # approx seconds/step, used only to scale the (capped) time bonus
N_AHEAD = 8               # how many lookahead waypoints the policy sees (richer corner preview)
GAP = 3                   # spacing between lookahead points, in waypoint indices
SEARCH_WINDOW = 20        # forward window for nearest-waypoint search (anti-shortcut)


class WaypointObs(gym.Wrapper):
    """Track-relative waypoint-following observation + checkpoint-gated progress reward."""

    def __init__(self, env: gym.Env, waypoints: np.ndarray, max_cte: float = 8.0,
                 n_ahead: int = N_AHEAD, gap: int = GAP, search_window: int = SEARCH_WINDOW,
                 num_checkpoints: int = NUM_CHECKPOINTS):
        super().__init__(env)
        self.wp = np.asarray(waypoints, dtype=np.float32)
        self.n = len(self.wp)
        self.max_cte = float(max_cte)
        self.n_ahead = int(n_ahead)
        self.gap = int(gap)
        self.window = int(search_window)
        self.num_cp = int(num_checkpoints)
        seg = np.linalg.norm(np.diff(np.vstack([self.wp, self.wp[0]]), axis=0), axis=1)
        self.spacing = float(seg.mean())
        self.look_scale = max(self.n_ahead * self.gap * self.spacing, 1.0)

        # progress / lap bookkeeping (initialised in reset)
        self._idx = 0
        self._cum = 0.0          # cumulative forward index-advancement this lap
        self._sectors = 0        # checkpoints crossed this lap (in order)
        self._step = 0
        self._lap_start_step = 0
        self._laps = 0
        self._last_action = np.zeros(2, dtype=np.float32)

        dim = 5 + 2 * self.n_ahead
        high = np.full(dim, 3.0, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def _vec(self, info: dict) -> np.ndarray:
        p = ground_xz(info.get("pos", (0.0, 0.0, 0.0)))
        car = info.get("car", (0.0, 0.0, 0.0))
        yaw = float(car[2]) if len(car) >= 3 else 0.0
        look = lookahead_local(self.wp, p, yaw, self._idx, self.n_ahead, self.gap)
        head = np.array(
            [
                signed_cte(self.wp, p, self._idx) / self.max_cte,
                float(info.get("speed", 0.0)) / 30.0,
                float(info.get("forward_vel", 0.0)) / 30.0,
                float(self._last_action[0]),
                float(self._last_action[1]),
            ],
            dtype=np.float32,
        )
        return np.concatenate([head, (look / self.look_scale).reshape(-1)]).astype(np.float32)

    def reset(self, **kwargs):
        self._last_action = np.zeros(2, dtype=np.float32)
        obs, info = self.env.reset(**kwargs)
        p = ground_xz(info.get("pos", (0.0, 0.0, 0.0)))
        self._idx = nearest_index(self.wp, p, 0, self.n)   # global nearest at episode start
        self._cum = 0.0
        self._sectors = 0
        self._step = 0
        self._lap_start_step = 0
        self._laps = 0
        return self._vec(info), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        cur_action = np.zeros(2, dtype=np.float32)
        cur_action[: min(2, a.shape[0])] = a[:2]
        # how much the action changed since last step (for the smoothness penalty)
        action_delta = float(np.abs(cur_action - self._last_action).sum())
        self._last_action = cur_action

        obs, _env_reward, terminated, truncated, info = self.env.step(action)
        self._step += 1

        p = ground_xz(info.get("pos", (0.0, 0.0, 0.0)))
        new_idx = nearest_index(self.wp, p, self._idx, self.window)
        advanced = (new_idx - self._idx) % self.n   # forward-only: in [0, window)
        self._idx = new_idx
        self._cum += advanced

        reward = K_PROGRESS * advanced * self.spacing
        reward += K_SPEED * float(info.get("forward_vel", 0.0))
        reward += TIME_PENALTY                       # anti-crawl time cost
        reward -= K_SMOOTH * action_delta            # anti-jitter smoothness penalty

        # ordered, once-per-lap checkpoint (sector) bonuses
        cp_size = self.n / self.num_cp
        while self._sectors < self.num_cp and self._cum >= (self._sectors + 1) * cp_size:
            reward += SECTOR_BONUS
            self._sectors += 1

        # lap completes only after a full loop of real progress (all checkpoints passed)
        if self._cum >= self.n:
            self._cum -= self.n
            self._sectors = 0
            self._laps += 1
            lap_steps = max(self._step - self._lap_start_step, 1)
            self._lap_start_step = self._step
            lap_seconds = max(lap_steps * SIM_DT, 1.0)
            reward += LAP_BONUS_FLAT + min(LAP_BONUS_TIME / lap_seconds, LAP_TIME_CAP)

        if terminated or truncated:
            reward += TERMINAL_PENALTY

        info["progress_frac"] = self._idx / self.n
        info["sectors"] = self._sectors
        info["laps"] = self._laps
        return self._vec(info), float(reward), terminated, truncated, info
