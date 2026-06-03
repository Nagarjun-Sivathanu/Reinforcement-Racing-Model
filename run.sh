#!/usr/bin/env bash
# Launcher for the Rl_Race SAC trainer. Uses this project's own venv (./.venv).
#
# Usage:
#   ./run.sh --timesteps 2000                    # smoke test
#   ./run.sh --timesteps 300000                  # full training run
#   ./run.sh --test --load models/sac_donkey     # watch a trained agent
#
# Prereq: start DonkeySim FIRST (this connects to a running sim; it does not launch one).
# The sim host is resolved as: $SIM_HOST, else the WSL2 gateway (Windows host), else localhost.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"

PORT="${SIM_PORT:-9091}"
HOST="${SIM_HOST:-$(ip route 2>/dev/null | awk '/^default/ {print $3; exit}')}"
HOST="${HOST:-127.0.0.1}"

if ! "$PY" - "$HOST" "$PORT" <<'PYEOF'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket(); s.settimeout(2)
try:
    s.connect((host, port)); print(f"[run.sh] Sim reachable at {host}:{port}")
except Exception as e:
    print(f"[run.sh] Cannot reach sim at {host}:{port} -> {e}"); sys.exit(1)
finally:
    s.close()
PYEOF
then
    echo "[run.sh] Is DonkeySim running and reachable? Start it (or set SIM_HOST), then re-run."
    exit 1
fi

exec "$PY" train_rl.py "$@"
