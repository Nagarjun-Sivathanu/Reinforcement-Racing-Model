"""Diagnostic test-drive: run a saved model and report WHERE and WHY it fails.

`train_rl.py --test` only prints per-episode totals, which tells us a lap happened but not
what went wrong. This drives the same policy and records per-step telemetry, then reports:

  * end reason per episode (wall collision vs off-track vs still-running)
  * the waypoint index where each episode died, bucketed into the 8 reward sectors
  * speed profile per sector (so we can see it arriving at a corner too fast)
  * lap times, both our own (progress-counted) and the sim's start-line timer

The model's own observation/action spaces decide `n_ahead` and `throttle_min`, so there is
no N_AHEAD env var to remember -- loading a 15-dim champ model and a 21-dim Path B model
both Just Work.

Usage:
  ./.venv/bin/python eval_model.py --load checkpoints/champ_continue3/best/best --episodes 6
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import pickle
import uuid
import zipfile

import numpy as np
import gymnasium as gym
import gym_donkeycar  # noqa: F401  (registers the donkey-* envs)
from stable_baselines3 import SAC

from reward import WaypointObs, NUM_CHECKPOINTS
from train_rl import sim_host, _noop_reward, MAX_CTE, ENV_NAME, PORT

HERE = os.path.dirname(os.path.abspath(__file__))


def model_spaces(path: str) -> tuple[int, float]:
    """(obs_dim, throttle_min) read straight out of the saved zip, without building an env."""
    zip_path = path if path.endswith(".zip") else path + ".zip"
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read("data").decode())
    obs = pickle.loads(base64.b64decode(data["observation_space"][":serialized:"]))
    act = pickle.loads(base64.b64decode(data["action_space"][":serialized:"]))
    return int(obs.shape[0]), float(act.low[1])


def build_env(env_id: str, waypoints: np.ndarray, n_ahead: int, throttle_min: float) -> gym.Env:
    conf = {
        "exe_path": "remote", "host": sim_host(), "port": PORT,
        "body_style": "f1", "body_rgb": (128, 128, 128), "car_name": "EVAL",
        "font_size": 100, "racer_name": "SAC-eval", "country": "Place",
        "bio": "diagnostic run", "guid": str(uuid.uuid4()),
        "max_cte": MAX_CTE, "steer_limit": 1.0,
        "throttle_min": throttle_min, "throttle_max": 1.0,
    }
    env = gym.make(env_id, conf=conf)
    env.unwrapped.set_reward_fn(_noop_reward)
    return WaypointObs(env, waypoints, max_cte=MAX_CTE, n_ahead=n_ahead)


def end_reason(info: dict) -> str:
    hit = str(info.get("hit", "none"))
    if hit not in ("none", "None", ""):
        return f"collision({hit})"
    if abs(float(info.get("cte", 0.0))) >= MAX_CTE:
        return "off-track(cte)"
    return "other"


def bar(value: float, peak: float, width: int = 28) -> str:
    n = 0 if peak <= 0 else int(round(width * value / peak))
    return "#" * n


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnostic test-drive of a saved model")
    p.add_argument("--load", required=True, help="path to a saved model (with or without .zip)")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--track", type=str, default="minimonaco")
    p.add_argument("--env", type=str, default=ENV_NAME)
    p.add_argument("--stochastic", action="store_true",
                   help="sample from the policy instead of driving its mean -- this is what "
                        "TRAINING sees, and is usually much worse than the deterministic run")
    p.add_argument("--max-steps", type=int, default=1500, help="per-episode step cap")
    p.add_argument("--csv", type=str, default=None, help="write per-step telemetry here")
    args = p.parse_args()

    waypoints = np.load(os.path.join(HERE, "tracks", f"{args.track}.npy"))
    n_wp = len(waypoints)
    obs_dim, throttle_min = model_spaces(args.load)
    n_ahead = (obs_dim - 5) // 2
    print(f"[eval] {args.load}: obs {obs_dim}-dim -> n_ahead={n_ahead} | "
          f"throttle range [{throttle_min}, 1.0] | track '{args.track}' ({n_wp} waypoints) | "
          f"{'stochastic' if args.stochastic else 'deterministic'}")

    env = build_env(args.env, waypoints, n_ahead, throttle_min)
    model = SAC.load(args.load, env=None, device="cpu")

    rows: list[dict] = []
    episodes: list[dict] = []
    sector_speed = [[] for _ in range(NUM_CHECKPOINTS)]   # speed samples per sector
    deaths = np.zeros(NUM_CHECKPOINTS, dtype=int)         # which sector each episode died in

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        total_r, steps, max_prog, lap_marks = 0.0, 0, 0.0, []
        speeds: list[float] = []
        while steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_r += reward
            idx = int(env._idx)
            sector = min(int(idx / n_wp * NUM_CHECKPOINTS), NUM_CHECKPOINTS - 1)
            speed = float(info.get("speed", 0.0))
            speeds.append(speed)
            sector_speed[sector].append(speed)
            max_prog = max(max_prog, info.get("progress_frac", 0.0))
            rows.append({
                "episode": ep, "step": steps, "wp_idx": idx, "sector": sector,
                "speed": round(speed, 3), "forward_vel": round(float(info.get("forward_vel", 0.0)), 3),
                "steer": round(float(action[0]), 3), "throttle": round(float(action[1]), 3),
                "cte": round(float(info.get("cte", 0.0)), 3), "reward": round(float(reward), 3),
                "laps": int(info.get("laps", 0)),
            })
            if int(info.get("laps", 0)) > len(lap_marks):
                lap_marks.append(steps)
            if terminated or truncated:
                break

        idx = int(env._idx)
        sector = min(int(idx / n_wp * NUM_CHECKPOINTS), NUM_CHECKPOINTS - 1)
        deaths[sector] += 1
        laps = int(info.get("laps", 0))
        # our own lap times, from the step at which each progress-lap completed
        our_laps = [round((b - a) * 0.05, 2)
                    for a, b in zip([0] + lap_marks[:-1], lap_marks)]
        episodes.append({
            "ep": ep, "steps": steps, "reward": total_r, "laps": laps,
            "end_wp": idx, "sector": sector, "reason": end_reason(info),
            "end_speed": speeds[-1] if speeds else 0.0,
            "cte": abs(float(info.get("cte", 0.0))),
            "sim_lap": float(info.get("last_lap_time", 0.0)), "our_laps": our_laps,
        })
        e = episodes[-1]
        print(f"[eval] ep {ep:>2}: {steps:>4} steps | reward {total_r:7.1f} | laps {laps} | "
              f"died at wp {idx:>2}/{n_wp} (sector {sector}) | {e['reason']} | "
              f"speed@end {e['end_speed']:5.2f} | |cte| {e['cte']:.2f} | our laps {our_laps}")

    env.close()

    # ---- summary ----------------------------------------------------------
    n = len(episodes)
    lapped = sum(1 for e in episodes if e["laps"] >= 1)
    all_laps = [t for e in episodes for t in e["our_laps"]]
    print("\n" + "=" * 72)
    print(f"SUMMARY over {n} episodes")
    print(f"  completed >=1 lap : {lapped}/{n} ({100*lapped/n:.0f}%)")
    if all_laps:
        print(f"  lap times (ours)  : n={len(all_laps)} best {min(all_laps):.2f}s  "
              f"mean {np.mean(all_laps):.2f}s  worst {max(all_laps):.2f}s")
    print(f"  mean steps/episode: {np.mean([e['steps'] for e in episodes]):.0f}")
    print(f"  mean reward       : {np.mean([e['reward'] for e in episodes]):.1f}")

    print("\n  WHERE IT DIES (sector = eighth of the lap, 0 starts at the start line):")
    peak = max(deaths.max(), 1)
    for s in range(NUM_CHECKPOINTS):
        lo, hi = int(s * n_wp / NUM_CHECKPOINTS), int((s + 1) * n_wp / NUM_CHECKPOINTS) - 1
        spd = np.mean(sector_speed[s]) if sector_speed[s] else 0.0
        print(f"    sector {s} (wp {lo:>2}-{hi:>2})  deaths {deaths[s]:>2} {bar(deaths[s], peak):<28} "
              f"mean speed {spd:5.2f}")

    reasons: dict[str, int] = {}
    for e in episodes:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    print("\n  WHY IT DIES:")
    for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {r:<22} {c}/{n}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  per-step telemetry -> {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
