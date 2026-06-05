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
import datetime
import os
import shutil
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


def build_model(env: gym.Env, buffer_size: int = 50000, learning_rate: float = 3e-4,
                learning_starts: int = 1000) -> SAC:
    """Fresh SAC. Bigger buffer_size (cheap with low-dim obs) = more stable, more diverse
    experience; lower learning_rate = steadier updates (anti-regression)."""
    return SAC(
        "MlpPolicy", env, verbose=1,
        buffer_size=buffer_size, learning_starts=learning_starts, batch_size=256,
        learning_rate=learning_rate, train_freq=64, gradient_steps=64,
        device="cpu", tensorboard_log=TB_DIR,
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
    p.add_argument("--run-name", type=str, default=None,
                   help="name for this run's checkpoint folder (default: timestamped). "
                        "Each run gets its own folder so models are never overwritten.")
    p.add_argument("--buffer-size", type=int, default=50000,
                   help="SAC replay buffer size (bigger = more stable; cheap with low-dim obs)")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate (lower = steadier)")
    p.add_argument("--learning-starts", type=int, default=1000,
                   help="random steps to seed the buffer before learning")
    p.add_argument("--load-buffer", type=str, default=None,
                   help="path to a saved replay buffer (.pkl) to resume WITHOUT the warm-start "
                        "dip (the 'look-back data'). Saved automatically on stop as "
                        "models/<run-name>_replay.pkl")
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
                      f"reached {ep_max_prog*100:5.1f}% of lap | laps {int(info.get('laps', 0))} | "
                      f"ended |cte|={abs(float(info.get('cte', 0.0))):.2f}")
                ep += 1
                ep_reward, ep_steps, ep_max_prog = 0.0, 0, 0.0
                obs, info = env.reset()
        env.close()
        return

    # Per-run checkpoint folder so a new run NEVER overwrites an older run's models.
    run_name = args.run_name or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(HERE, "checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)
    print(f"[train_rl] run '{run_name}' -> checkpoints in {ckpt_dir}")

    if args.load:
        # Always warm-start WEIGHTS into a fresh model built with the CLI hyperparameters, so
        # stability settings (buffer_size, lr) actually apply (SAC.load would restore the old
        # saved hyperparameters instead). Network shapes must match the loaded model: keep the
        # same obs dim (N_AHEAD) and action dim. Works for same-space continues AND action-space
        # changes (e.g. adding braking) since the action is still 2-D.
        print(f"[train_rl] warm-starting weights from {args.load} "
              f"(buffer={args.buffer_size} lr={args.lr})")
        model = build_model(env, args.buffer_size, args.lr, args.learning_starts)
        model.set_parameters(args.load)
        # Restoring the replay buffer ("look-back data") avoids the warm-start dip entirely:
        # the critic keeps its experience instead of relearning from an empty buffer.
        if args.load_buffer and os.path.exists(args.load_buffer):
            model.load_replay_buffer(args.load_buffer)
            print(f"[train_rl] loaded replay buffer from {args.load_buffer} "
                  f"({model.replay_buffer.size()} samples) - resuming WITHOUT a dip")
        elif args.load_buffer:
            print(f"[train_rl] WARNING: --load-buffer {args.load_buffer} not found; "
                  f"resuming with empty buffer (will dip)")
    else:
        print(f"[train_rl] new SAC model (MlpPolicy, CPU) buffer={args.buffer_size} lr={args.lr}")
        model = build_model(env, args.buffer_size, args.lr, args.learning_starts)

    ckpt = CheckpointCallback(
        save_freq=5000,
        save_path=ckpt_dir,
        name_prefix="sac_donkey",
    )
    print(f"[train_rl] training for {args.timesteps} timesteps (Ctrl-C to stop & save)")
    try:
        model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=True)
    except KeyboardInterrupt:
        print("\n[train_rl] interrupted - saving current model")
    # Save a per-run final model (never overwritten) plus a stable convenience copy.
    final_run = os.path.join(HERE, "models", f"{run_name}.zip")
    model.save(final_run)
    shutil.copyfile(final_run, os.path.join(HERE, "models", "sac_donkey.zip"))
    # Save the replay buffer ("look-back data") so a later resume can skip the warm-start dip:
    #   ./run.sh --load models/<run-name> --load-buffer models/<run-name>_replay.pkl ...
    buf_path = os.path.join(HERE, "models", f"{run_name}_replay.pkl")
    try:
        model.save_replay_buffer(buf_path)
        print(f"[train_rl] saved replay buffer -> {buf_path} (use --load-buffer to resume dip-free)")
    except Exception as e:
        print(f"[train_rl] could not save replay buffer: {e}")
    print(f"[train_rl] saved -> {final_run} (and copied to models/sac_donkey.zip)")
    env.close()


if __name__ == "__main__":
    main()
