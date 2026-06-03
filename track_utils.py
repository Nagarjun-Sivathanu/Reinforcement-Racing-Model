"""Track-geometry helpers shared by the recorder, the observation wrapper, and training.

The whole point of these utilities is *track-agnostic* path following: a track is just an
ordered list of centerline points on the ground plane (x, z). Given the car's position and
heading, we can compute where it is along the path (progress), how far off the path it is
(signed cross-track error), and what the road looks like *ahead* in the car's own frame.
Feeding that "road ahead" into the policy is what lets one trained brain drive a *different*
track by swapping in that track's points.

Coordinate convention (Unity / DonkeySim): Y is up; the ground plane is (x, z); `yaw` is the
rotation about Y in DEGREES, and the car's forward ground vector is (sin yaw, cos yaw).
"""
from __future__ import annotations

import numpy as np


def ground_xz(pos) -> np.ndarray:
    """(pos_x, pos_y, pos_z) -> ground point (x, z) as float32."""
    return np.array([float(pos[0]), float(pos[2])], dtype=np.float32)


def yaw_to_forward(yaw_deg: float) -> np.ndarray:
    """Car forward unit vector on the ground plane, from yaw in degrees."""
    r = np.radians(float(yaw_deg))
    return np.array([np.sin(r), np.cos(r)], dtype=np.float32)


def resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a closed polyline to ~evenly spaced points (by arc length).

    `points` is (N, 2) raw recorded path; returns (M, 2) evenly spaced and de-duplicated,
    treated as a closed loop (last connects back to first). Even spacing matters so that
    "advance one waypoint" means the same physical distance everywhere on the track.
    """
    pts = np.asarray(points, dtype=np.float64)
    # drop consecutive duplicates (the car sits still at start/resets)
    keep = np.concatenate(([True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-3))
    pts = pts[keep]
    closed = np.vstack([pts, pts[0]])                       # close the loop
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))             # arc length at each vertex
    total = float(s[-1])
    if total < spacing * 3:
        raise ValueError(f"path too short ({total:.2f}) for spacing {spacing}")
    targets = np.arange(0.0, total, spacing)
    out_x = np.interp(targets, s, closed[:, 0])
    out_z = np.interp(targets, s, closed[:, 1])
    return np.stack([out_x, out_z], axis=1).astype(np.float32)


def nearest_index(waypoints: np.ndarray, p: np.ndarray, last_idx: int, window: int) -> int:
    """Index of the closest waypoint, searching only a forward window from `last_idx`.

    Restricting the search forward keeps "progress" monotonic and stops the agent from
    teleporting its progress by cutting across the track (shortcut exploitation). The window
    wraps around the closed loop.
    """
    n = len(waypoints)
    idxs = (last_idx + np.arange(0, window)) % n
    d = np.linalg.norm(waypoints[idxs] - p[None, :], axis=1)
    return int(idxs[int(np.argmin(d))])


def signed_cte(waypoints: np.ndarray, p: np.ndarray, idx: int) -> float:
    """Signed lateral distance from the path at waypoint `idx` (left +, right -).

    Uses the local path direction (idx -> idx+1) and the cross product with the car offset.
    """
    n = len(waypoints)
    a = waypoints[idx]
    b = waypoints[(idx + 1) % n]
    tangent = b - a
    norm = np.linalg.norm(tangent)
    if norm < 1e-6:
        return 0.0
    tangent = tangent / norm
    off = p - a
    # 2D cross product (tangent x offset): sign gives which side of the path.
    return float(tangent[0] * off[1] - tangent[1] * off[0])


def lookahead_local(
    waypoints: np.ndarray, p: np.ndarray, yaw_deg: float, idx: int,
    n_ahead: int, gap: int,
) -> np.ndarray:
    """The next `n_ahead` waypoints (every `gap`-th), expressed in the car's local frame.

    Local frame: x = lateral (right), y = forward. This is the track-agnostic description of
    "the road ahead" the policy steers by. Returns shape (n_ahead, 2).
    """
    n = len(waypoints)
    fwd = yaw_to_forward(yaw_deg)              # (sin, cos)
    right = np.array([fwd[1], -fwd[0]], dtype=np.float32)   # forward rotated -90deg
    out = np.zeros((n_ahead, 2), dtype=np.float32)
    for k in range(n_ahead):
        wp = waypoints[(idx + (k + 1) * gap) % n]
        rel = wp - p
        out[k, 0] = float(np.dot(rel, right))   # lateral offset
        out[k, 1] = float(np.dot(rel, fwd))     # forward offset
    return out
