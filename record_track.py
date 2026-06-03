"""Record a track's centerline by gently driving one lap, then save evenly spaced waypoints.

This produces the per-track "points" you swap to deploy the same policy on a different track.
It drives a PD-controller on cross-track error (steer toward center, low throttle), logs the
car's ground path (x, z), and on lap completion resamples to even spacing and saves to
tracks/<name>.npy.

Lap completion is detected GEOMETRICALLY: the car must travel far from its start, then return
near it, after covering enough distance. We do NOT trust the sim's start-line lap counter --
weaving over the start trigger near spawn can trip it without a real lap (this produced a
tangled, partial "lap" before). After capture, the loop is VALIDATED (total turning ~360deg)
and only then saved.

Prereq: start DonkeySim.exe on Windows first (this connects; it does not launch one).

Usage:
  python record_track.py --name minimonaco --steer-sign -1
  python record_track.py --name warren --env donkey-warren-track-v0 --steer-sign -1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid

import numpy as np
import gymnasium as gym
import gym_donkeycar  # noqa: F401

from track_utils import ground_xz, resample_closed

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("SIM_PORT", "9091"))
MAX_CTE = 8.0


def sim_host() -> str:
    """Host where DonkeySim runs: $SIM_HOST, else the WSL2 gateway, else localhost."""
    env_host = os.environ.get("SIM_HOST")
    if env_host:
        return env_host
    try:
        out = subprocess.check_output(["ip", "route"]).decode()
        for line in out.splitlines():
            if line.startswith("default"):
                return line.split()[2]
    except Exception:
        pass
    return "127.0.0.1"


def make_env(env_name: str) -> gym.Env:
    conf = {
        "exe_path": "remote", "host": sim_host(), "port": PORT,
        "body_style": "f1", "body_rgb": (200, 200, 200), "car_name": "REC",
        "font_size": 100, "racer_name": "recorder", "country": "Place",
        "bio": "recording centerline", "guid": str(uuid.uuid4()),
        "max_cte": MAX_CTE, "steer_limit": 1.0, "throttle_min": 0.0, "throttle_max": 1.0,
    }
    return gym.make(env_name, conf=conf)


def signed_turning_deg(w: np.ndarray) -> float:
    """Sum of signed turn angles around the closed loop. One simple loop = +/-360."""
    v = np.diff(np.vstack([w, w[0]]), axis=0)
    ang = np.arctan2(v[:, 1], v[:, 0])
    dt = (np.diff(np.concatenate([ang, [ang[0]]])) + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(dt.sum()))


def ascii_preview(w: np.ndarray, width: int = 56, height: int = 24) -> str:
    xs, zs = w[:, 0], w[:, 1]
    gx = ((xs - xs.min()) / max(np.ptp(xs), 1e-6) * (width - 1)).astype(int)
    gz = ((zs - zs.min()) / max(np.ptp(zs), 1e-6) * (height - 1)).astype(int)
    grid = [[" "] * width for _ in range(height)]
    for i, (a, b) in enumerate(zip(gx, gz)):
        grid[height - 1 - b][a] = "0" if i == 0 else ("*" if i == len(w) - 1 else ".")
    return "\n".join("".join(r) for r in grid)


def _pd_step(env, info, kp, kd, steer_sign, throttle, prev_cte):
    cte = float(info.get("cte", 0.0))
    d_cte = cte - prev_cte
    steer = float(np.clip(steer_sign * (kp * cte + kd * d_cte), -1.0, 1.0))
    obs, _r, term, trunc, info = env.step(np.array([steer, throttle], dtype=np.float32))
    return info, term, trunc, cte


def record_lap(env, kp, kd, steer_sign, throttle, max_steps, attempts,
               settle=20, far_thresh=5.0, close_radius=2.0, min_dist=25.0) -> np.ndarray:
    """Drive a PD-controller and return the raw (N,2) path of one geometrically-verified lap."""
    for attempt in range(1, attempts + 1):
        obs, info = env.reset()
        prev_cte = float(info.get("cte", 0.0))
        # settle: leave the spawn before we fix the start reference
        off = False
        for _ in range(settle):
            info, term, trunc, prev_cte = _pd_step(env, info, kp, kd, steer_sign, throttle, prev_cte)
            if term or trunc:
                off = True
                break
        if off:
            print(f"[record] attempt {attempt}: went off during settle - retrying")
            continue

        p0 = ground_xz(info.get("pos", (0.0, 0.0, 0.0)))
        path = [tuple(p0)]
        cum = 0.0
        max_d = 0.0
        been_far = False
        prev_p = p0
        cte_max = 0.0
        for step in range(max_steps):
            info, term, trunc, last_cte = _pd_step(env, info, kp, kd, steer_sign, throttle, prev_cte)
            prev_cte = last_cte   # derivative = next observed cte - cte we just steered on
            cte_max = max(cte_max, abs(last_cte))
            p = ground_xz(info.get("pos", (0.0, 0.0, 0.0)))
            path.append(tuple(p))
            cum += float(np.linalg.norm(p - prev_p))
            prev_p = p
            d0 = float(np.linalg.norm(p - p0))
            max_d = max(max_d, d0)
            if d0 > far_thresh:
                been_far = True
            if been_far and d0 < close_radius and cum > min_dist:
                print(f"[record] attempt {attempt}: LAP verified at step {step} "
                      f"(path_len={cum:.1f}, max_dist_from_start={max_d:.1f}, |cte|max={cte_max:.2f})")
                return np.array(path, dtype=np.float32)
            if term or trunc:
                print(f"[record] attempt {attempt}: off-track at step {step} "
                      f"(got {cum:.1f} units, max_dist={max_d:.1f}) - retrying")
                break
        else:
            print(f"[record] attempt {attempt}: max_steps without returning to start "
                  f"(path_len={cum:.1f}, max_dist={max_d:.1f}) - retrying")
    raise RuntimeError("Could not record a verified lap. Adjust --kp/--kd/--throttle, or "
                       "flip --steer-sign. The car must drive a full loop back to start.")


def main():
    p = argparse.ArgumentParser(description="Record a verified track centerline")
    p.add_argument("--name", required=True, help="track name -> tracks/<name>.npy")
    p.add_argument("--env", default="donkey-minimonaco-track-v0")
    p.add_argument("--kp", type=float, default=0.6)
    p.add_argument("--kd", type=float, default=2.5)
    p.add_argument("--steer-sign", type=float, default=-1.0, help="flip to +1 if it diverges")
    p.add_argument("--throttle", type=float, default=0.2)
    p.add_argument("--spacing", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--attempts", type=int, default=5)
    args = p.parse_args()

    env = make_env(args.env)
    try:
        raw = record_lap(env, args.kp, args.kd, args.steer_sign, args.throttle,
                         args.max_steps, args.attempts)
    finally:
        env.close()

    waypoints = resample_closed(raw, args.spacing)
    arc = float(np.sum(np.linalg.norm(np.diff(np.vstack([waypoints, waypoints[0]]), axis=0), axis=1)))
    turning = signed_turning_deg(waypoints)
    bb_lo, bb_hi = waypoints.min(0), waypoints.max(0)

    print(f"\n[record] {len(raw)} raw pts -> {len(waypoints)} waypoints, lap length ~{arc:.1f}")
    print(f"[record] total signed turning = {turning:.0f} deg (a clean single loop ~ +/-360)")
    print(f"[record] bbox x:[{bb_lo[0]:.1f},{bb_hi[0]:.1f}] z:[{bb_lo[1]:.1f},{bb_hi[1]:.1f}]\n")
    print(ascii_preview(waypoints))

    # VALIDATION GATE: refuse to save a tangled / partial capture.
    if not (270.0 <= abs(turning) <= 450.0):
        print(f"\n[record] REJECTED: turning {turning:.0f} deg is not ~360 -> not a clean single "
              f"lap. Nothing saved. Re-run (watch the sim; adjust --throttle/--kp/--kd).")
        sys.exit(1)

    os.makedirs(os.path.join(HERE, "tracks"), exist_ok=True)
    out = os.path.join(HERE, "tracks", f"{args.name}.npy")
    np.save(out, waypoints)
    np.save(os.path.join(HERE, "tracks", f"{args.name}_raw.npy"), raw)  # keep raw for debugging
    print(f"\n[record] VALIDATED & saved -> {out}")


if __name__ == "__main__":
    main()
