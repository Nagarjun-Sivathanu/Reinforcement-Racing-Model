"""Waypoint-following observation + progress reward for track-agnostic DonkeySim RL.

Design (agreed with the user): the policy must *see the road ahead* so that one trained brain
can drive a different track by swapping in that track's waypoints. So the observation is built
in the car's own frame from a per-track centerline (see track_utils + record_track.py):

    [ signed_cte, speed, forward_vel, last_steer, last_throttle,
      (lateral, forward) of the next N lookahead waypoints ... ]

Everything is track-relative, so it transfers across tracks. The reward is dense *progress
along the path* (potential-based: rewards arc-length advanced, not a moving best-time
baseline, which would be non-stationary and destabilize training), plus a speed term, a
terminal penalty, and a faster-lap bonus. There is deliberately no centering term -- the
racing line is free to leave the exact center; staying on track is enforced by the env's
max_cte termination.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from track_utils import ground_xz, nearest_index, signed_cte, lookahead_local

# --- reward weights (tune these) -------------------------------------------
K_PROGRESS = 1.0          # reward per world-unit advanced along the path (the workhorse)
K_SPEED = 0.05            # small bonus per unit forward velocity (encourages commitment)
TERMINAL_PENALTY = -10.0  # ending the episode (off-track / crash)
LAP_BONUS_FLAT = 50.0     # flat reward for completing a lap
LAP_BONUS_TIME = 300.0    # added as LAP_BONUS_TIME / lap_time -> faster lap, bigger bonus

# --- observation shape -----------------------------------------------------
N_AHEAD = 5               # how many lookahead waypoints the policy sees
GAP = 3                   # spacing between lookahead points, in waypoint indices
SEARCH_WINDOW = 20        # forward window for nearest-waypoint search (anti-shortcut)


class WaypointObs(gym.Wrapper):
    """Replace the image observation with a track-relative waypoint-following vector, and
    compute the progress reward here (where the per-track waypoints live)."""

    def __init__(self, env: gym.Env, waypoints: np.ndarray, max_cte: float = 8.0,
                 n_ahead: int = N_AHEAD, gap: int = GAP, search_window: int = SEARCH_WINDOW):
        super().__init__(env)
        self.wp = np.asarray(waypoints, dtype=np.float32)
        self.n = len(self.wp)
        self.max_cte = float(max_cte)
        self.n_ahead = int(n_ahead)
        self.gap = int(gap)
        self.window = int(search_window)
        # mean segment length -> convert "indices advanced" to world distance for the reward
        seg = np.linalg.norm(np.diff(np.vstack([self.wp, self.wp[0]]), axis=0), axis=1)
        self.spacing = float(seg.mean())
        # scale for normalizing lookahead coords (~ farthest lookahead distance)
        self.look_scale = max(self.n_ahead * self.gap * self.spacing, 1.0)

        self._idx = 0
        self._prev_lap = 0
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
        # global nearest at episode start (search the whole loop)
        self._idx = nearest_index(self.wp, p, 0, self.n)
        self._prev_lap = int(info.get("lap_count", 0))
        return self._vec(info), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_action = np.zeros(2, dtype=np.float32)
        self._last_action[: min(2, a.shape[0])] = a[:2]

        obs, _env_reward, terminated, truncated, info = self.env.step(action)

        p = ground_xz(info.get("pos", (0.0, 0.0, 0.0)))
        new_idx = nearest_index(self.wp, p, self._idx, self.window)
        advanced = (new_idx - self._idx) % self.n   # always in [0, window): forward-only
        self._idx = new_idx

        reward = K_PROGRESS * advanced * self.spacing
        reward += K_SPEED * float(info.get("forward_vel", 0.0))

        cur_lap = int(info.get("lap_count", 0))
        if cur_lap > self._prev_lap:
            lap_time = float(info.get("last_lap_time", 0.0) or 0.0)
            reward += LAP_BONUS_FLAT
            if lap_time > 0:
                reward += LAP_BONUS_TIME / lap_time
        self._prev_lap = cur_lap

        if terminated or truncated:
            reward += TERMINAL_PENALTY

        # expose progress so evaluation/logging can report how far around the lap we got
        info["progress_frac"] = self._idx / self.n
        info["progress_idx"] = self._idx

        return self._vec(info), float(reward), terminated, truncated, info
