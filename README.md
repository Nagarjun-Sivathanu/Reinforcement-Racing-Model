# Rl_Race

**A reinforcement-learning racing agent for [DonkeySim](https://github.com/tawnkramer/gym-donkeycar), trained with Soft Actor-Critic (SAC) to learn its own racing line — not to copy a human.**

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue.svg">
  <img alt="RL" src="https://img.shields.io/badge/RL-Stable--Baselines3%20(SAC)-orange.svg">
  <img alt="Sim" src="https://img.shields.io/badge/sim-gym--donkeycar-green.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey.svg">
</p>

Rl_Race trains a car to drive — and ultimately to optimize lap time — on a track in the
DonkeyCar simulator. Unlike behaviour cloning, which is capped at the skill of the human it
imitates, the agent here discovers good driving from a reward signal and can, in principle,
exceed any demonstration. The design philosophy is *"DeepRacer at home"*: a single track, a
lightweight observation, and a hand-crafted dense reward, where **reward shaping — not the
algorithm — is the real work.**

---

## Highlights

- **Soft Actor-Critic** (Stable-Baselines3) for stable, sample-efficient continuous control
  (steering + throttle).
- **Waypoint-following design** — the agent observes the *track ahead* in its own reference
  frame, so a single trained policy can drive a **different track by swapping in that track's
  waypoints** (see [Generalization](#generalization)).
- **Low-dimensional observation, not camera frames** — a tiny state vector instead of images,
  which removes the multi-gigabyte image replay buffer and keeps memory flat (designed around
  a 16 GB machine; see [Hardware notes](#hardware-notes)).
- **Progress-based dense reward** with a faster-lap bonus — and *deliberately no centring
  term*, so the optimal racing line (which hugs apexes, not the middle) is free to emerge.
- **Automatic track capture** — a self-driving recorder traces and validates a track's
  centerline into reusable waypoints.
- **Reproducible** — pinned dependencies, a Dockerfile, and TensorBoard logging.

---

## How it works

### Observation — *"what the car sees"*
The raw camera image is replaced by a compact, **track-relative** state vector:

```
[ signed_cross_track_error,  speed,  forward_velocity,  last_steer,  last_throttle,
  (lateral, forward) offsets of the next N look-ahead waypoints ... ]
```

The look-ahead waypoints, expressed in the car's own frame, describe the shape of the road
ahead. Because this description is track-independent, the learned skill — *"given the road
ahead looks like this, steer like that"* — transfers to other tracks.

### Reward — *"what good driving means"*
Computed every step (`reward.py`):

| Term | Default | Fires | Purpose |
|------|--------:|-------|---------|
| `K_PROGRESS` | `1.0` | each step | Reward per unit of distance advanced **along the path**. The workhorse — a full lap is worth ~+70 from this alone. |
| `K_SPEED` | `0.05` | each step | A small bonus on forward velocity to encourage commitment. |
| `TERMINAL_PENALTY` | `-10.0` | episode end | Penalty for leaving the track / crashing. |
| `LAP_BONUS_FLAT` | `+50.0` | lap complete | Flat reward for finishing a lap. |
| `LAP_BONUS_TIME` | `+300 / lap_time` | lap complete | Faster lap → bigger bonus (the seed of lap-time optimization). |

Track limits are enforced by the environment's `max_cte` **episode termination** (not a
cross-track penalty), keeping the racing line unconstrained. Progress is measured with a
forward-only search window so the agent can't "teleport" progress by cutting across the track.

---

## Results

A 300k-step run on the Mini Monaco track (CPU, ~7 hours of real-time simulation) showed the
characteristic SAC learning curve — early exploration, a plateau, then a breakthrough:

| Training step | Mean episode reward | Mean episode length |
|--------------:|--------------------:|--------------------:|
| ~10k          | ~0                  | ~63                 |
| ~67k          | ~16                 | ~109                |
| ~150k         | ~20 (plateau)       | ~117                |
| ~281k         | **~67 (peak)**      | **~227**            |

Episode reward climbed from negative (a random policy that crashes immediately) to a peak
around **+67**, with episodes more than doubling in length as the car learned to stay on
track and drive progressively further around the lap. Checkpoints are saved every 5k steps,
so the best-performing checkpoint can be selected for deployment.

> Reproduce the curves: `tensorboard --logdir tb` and open <http://localhost:6006>.

---

## Repository structure

```
Rl_Race/
├── train_rl.py        # SAC train / test entry point; builds the env, wires the reward
├── reward.py          # WaypointObs (observation wrapper) + the progress reward
├── track_utils.py     # track geometry: nearest-waypoint, signed CTE, look-ahead in car frame
├── record_track.py    # self-driving recorder: traces & validates a track's centerline
├── run.sh             # venv launcher with a sim-reachability pre-check
├── requirements.txt   # pinned dependencies
├── Dockerfile         # containerized trainer
├── docker-compose.yml # convenience wrapper
└── tracks/            # recorded waypoints, e.g. minimonaco.npy
```

---

## Getting started

### Prerequisites
- **DonkeySim** running and reachable (download from the
  [gym-donkeycar releases](https://github.com/tawnkramer/gym-donkeycar/releases)). This project
  *connects* to a running sim; it does not launch one.
- Python 3.10.

### Installation (local)
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 1. Record a track's waypoints
A simple PD controller drives one gentle lap, then the centerline is validated (it must be a
clean ~360° loop) and saved:
```bash
python record_track.py --name minimonaco
```
> A repo-ready `tracks/minimonaco.npy` is already included.

### 2. Train
```bash
./run.sh --timesteps 300000                  # uses tracks/minimonaco.npy by default
# or directly:
python train_rl.py --timesteps 300000 --track minimonaco
```
Checkpoints land in `checkpoints/` every 5k steps; the final model saves to
`models/sac_donkey.zip`. `Ctrl-C` saves before exiting.

### 3. Watch a trained agent
```bash
./run.sh --test --load checkpoints/sac_donkey_280000_steps
```

### 4. Monitor training
```bash
tensorboard --logdir tb     # http://localhost:6006
```

---

## Docker

The container holds **only the trainer** — DonkeySim runs outside it. Point the container at
the sim with `SIM_HOST` / `SIM_PORT`.

```bash
docker build -t rl_race .

docker run --rm \
  -e SIM_HOST=<ip-of-machine-running-the-sim> \
  -v "$PWD/models:/app/models" \
  -v "$PWD/checkpoints:/app/checkpoints" \
  -v "$PWD/tb:/app/tb" \
  rl_race --timesteps 300000
```

Or with Compose:
```bash
SIM_HOST=<sim-ip> docker compose up --build
```
On Docker Desktop (Windows/Mac) use `SIM_HOST=host.docker.internal`; on native Linux prefer
`network_mode: host` with `SIM_HOST=127.0.0.1`.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SIM_HOST` | WSL2 gateway → `127.0.0.1` | Host running DonkeySim |
| `SIM_PORT` | `9091` | DonkeySim port |
| `SIM_ENV`  | `donkey-minimonaco-track-v0` | gym-donkeycar environment id |

Reward weights and observation shape (`N_AHEAD`, `GAP`, …) are constants at the top of
`reward.py`. Changes take effect on the **next** training run.

---

## Generalization

The recorded waypoints play two roles — *measuring progress* and *describing the road ahead
in the observation* — but they are **never a centring target**. Because the observation is
track-relative, the intended workflow is:

1. Train a policy on one track.
2. Record a new track's waypoints (`record_track.py --name <track> --env <env-id>`).
3. Deploy the **same policy** against the new track's waypoints.

Robust cross-track performance ultimately benefits from training on several tracks (domain
randomization); single-track-then-transfer is the first milestone.

---

## Hardware notes

The binding constraint is **RAM and simulator throughput, not the GPU**. RL policy networks
are tiny, so:
- **Torch runs on CPU on purpose** — the MLP policy is small and the real-time sim is the
  throughput bottleneck. This also avoids contending for GPU memory with the sim.
- **A low-dimensional observation** removes the image replay buffer entirely, so memory stays
  flat throughout training (verified across a full 300k-step run).

Expect **hours** of real-time simulation per run, plus several reward-tuning iterations — this
is a learning project, not a one-shot.

---

## Roadmap

- [ ] **Braking** — throttle is currently `[0, 1]` (accelerate/coast only); allowing negative
      throttle should help tight corners.
- [ ] **Best-model checkpointing** via `EvalCallback` (currently periodic + final only).
- [ ] **Multi-track training** for genuine generalization.
- [ ] Lap-time leaderboard / comparison against a behaviour-cloning baseline.

---

## Acknowledgements

- [gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar) — the DonkeySim Gym environment.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — the SAC implementation.

## License

Released under the MIT License — see [LICENSE](LICENSE).
