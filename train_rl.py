"""Train (or test) a SAC agent on DonkeySim.

Phase-2 RL agent: SAC + low-dimensional observation + the progress/track-limits/lap-time
reward from reward.py. Sized for a 16 GB / RTX 3060-Laptop box (MlpPolicy on CPU torch;
the real-time sim is the bottleneck, not the GPU).

Prereq: start DonkeySim.exe on Windows FIRST (this connects to it; it does not launch one).

Usage:
  python rl/train_rl.py --timesteps 2000          # smoke test
  python rl/train_rl.py --timesteps 200000        # real training run
  python rl/train_rl.py --test --load rl/models/sac_donkey   # watch a trained agent
"""
from __future__ import annotations

import argparse
import os
import subprocess
import uuid

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import numpy as np
import gym_donkeycar  # noqa: F401  (registers the donkey-* envs)

from reward import WaypointObs

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_NAME = os.environ.get("SIM_ENV", "donkey-minimonaco-track-v0")
PORT = int(os.environ.get("SIM_PORT", "9091"))
MAX_CTE = 8.0
TB_DIR = os.path.join(HERE, "tb")         # TensorBoard logs: `tensorboard --logdir tb`


def _noop_reward(self, done):  # reward is computed in WaypointObs; ignore the handler's
    return 0.0


def sim_host() -> str:
    """Host where DonkeySim runs. Priority: $SIM_HOST (set this in Docker / remote setups),
    then the WSL2 default-route gateway (the Windows host when running under WSL), then
    localhost. Resolved at runtime so it survives WSL restarts and works in containers."""
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


def make_env(env_name: str, waypoints: np.ndarray, throttle_min: float = 0.0) -> gym.Env:
    conf = {
        "exe_path": "remote",          # attach to the already-running sim
        "host": sim_host(),
        "port": PORT,
        "body_style": "f1",
        "body_rgb": (128, 128, 128),
        "car_name": "RL",
        "font_size": 100,
        "racer_name": "SAC",
        "country": "Place",
        "bio": "Learning to race with SAC",
        "guid": str(uuid.uuid4()),
        "max_cte": MAX_CTE,
        "steer_limit": 1.0,
        "throttle_min": throttle_min,   # set negative to allow braking
        "throttle_max": 1.0,
    }
    env = gym.make(env_name, conf=conf)
    env.unwrapped.set_reward_fn(_noop_reward)      # reward is computed in WaypointObs
    env = WaypointObs(env, waypoints, max_cte=MAX_CTE)  # track-relative waypoint following
    env = Monitor(env)                             # episode reward/length logging
    return env


def build_model(env: gym.Env) -> SAC:
    """Fresh SAC with the project's hyperparameters (small replay buffer -> low RAM)."""
    return SAC(
        "MlpPolicy", env, verbose=1,
        buffer_size=50000, learning_starts=1000, batch_size=256,
        train_freq=64, gradient_steps=64, device="cpu", tensorboard_log=TB_DIR,
    )


def load_waypoints(track: str) -> np.ndarray:
    path = os.path.join(HERE, "tracks", f"{track}.npy")
    if not os.path.exists(path):
        raise SystemExit(
            f"[train_rl] no waypoints at {path}. Record them first:\n"
            f"  ./run.sh-equivalent: python record_track.py --name {track}"
        )
    return np.load(path)


def main():
    p = argparse.ArgumentParser(description="SAC trainer for DonkeySim")
    p.add_argument("--timesteps", type=int, default=200000)
    p.add_argument("--test", action="store_true", help="load --load model and just drive")
    p.add_argument("--load", type=str, default=None, help="path to a saved model (no .zip)")
    p.add_argument("--track", type=str, default="minimonaco", help="waypoints: tracks/<track>.npy")
    p.add_argument("--env", type=str, default=ENV_NAME, help="gym-donkeycar env id")
    p.add_argument("--throttle-min", type=float, default=0.0,
                   help="min throttle; set negative (e.g. -1.0) to allow braking")
    args = p.parse_args()

    waypoints = load_waypoints(args.track)
    print(f"[train_rl] track '{args.track}': {len(waypoints)} waypoints, env {args.env}, "
          f"throttle_min={args.throttle_min}")
    env = make_env(args.env, waypoints, throttle_min=args.throttle_min)

    if args.test:
        model_path = args.load or os.path.join(HERE, "models", "sac_donkey")
        print(f"[train_rl] loading {model_path} and driving deterministically")
        model = SAC.load(model_path, env=env)
        obs, info = env.reset()
        ep, ep_reward, ep_steps, ep_max_prog = 1, 0.0, 0, 0.0
        for _ in range(5000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            ep_max_prog = max(ep_max_prog, info.get("progress_frac", 0.0))
            if terminated or truncated:
                print(f"[eval] episode {ep:>2}: {ep_steps:>4} steps | reward {ep_reward:7.1f} | "
                      f"reached {ep_max_prog*100:5.1f}% of lap | laps {int(info.get('lap_count', 0))} | "
                      f"ended |cte|={abs(float(info.get('cte', 0.0))):.2f}")
                ep += 1
                ep_reward, ep_steps, ep_max_prog = 0.0, 0, 0.0
                obs, info = env.reset()
        env.close()
        return

    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    if args.load:
        try:
            model = SAC.load(args.load, env=env, tensorboard_log=TB_DIR)
            print(f"[train_rl] continuing training from {args.load} (matched spaces)")
        except ValueError as e:
            # Spaces changed (e.g. throttle range widened for braking). Network shapes are
            # unchanged (action is still 2-D), so warm-start: copy weights into a fresh model
            # with the NEW action space. Steering/track knowledge transfers; throttle relearns.
            print(f"[train_rl] space change detected ({e}); warm-starting weights from {args.load}")
            model = build_model(env)
            model.set_parameters(args.load)
    else:
        print("[train_rl] new SAC model (MlpPolicy, CPU)")
        model = build_model(env)

    ckpt = CheckpointCallback(
        save_freq=5000,
        save_path=os.path.join(HERE, "checkpoints"),
        name_prefix="sac_donkey",
    )
    print(f"[train_rl] training for {args.timesteps} timesteps (Ctrl-C to stop & save)")
    try:
        model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=True)
    except KeyboardInterrupt:
        print("\n[train_rl] interrupted - saving current model")
    final = os.path.join(HERE, "models", "sac_donkey")
    model.save(final)
    print(f"[train_rl] saved -> {final}.zip")
    env.close()


if __name__ == "__main__":
    main()
