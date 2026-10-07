#!/usr/bin/env bash
# Detached logger + best-peak saver, generalised over run name.
#
# monitor_pathB.sh had the run's paths hard-coded, so every new run needed a hand-edit (and it
# still ended with a stale pathB_long path in its closing message). This takes the run name as
# an argument instead:
#
#   setsid ./monitor_run.sh champ_continue4 >> logs/monitor_champ4.out 2>&1 &
#
# Appends a status row to TRAINING_LOG.md every INTERVAL seconds and copies the newest
# checkpoint to <ckpt>/best/best.zip whenever ep_rew_mean sets a new all-time high for the run.
set -u
ROOT=/home/nagu/projects/Rl_Race
RUN="${1:?usage: monitor_run.sh <run-name> [logfile]}"
LOG="${2:-$ROOT/logs/train_${RUN}.log}"
MD=$ROOT/TRAINING_LOG.md
CKPT=$ROOT/checkpoints/$RUN
BEST=$CKPT/best
TOTAL=${TOTAL:-1000000}
INTERVAL=${INTERVAL:-300}
# No leading "--": pgrep would parse it as its own option and bail out with a usage message.
PATTERN="run-name $RUN"
mkdir -p "$BEST"
best="-1000000"

# Give the trainer a moment to come up before we start watching for it.
for _ in $(seq 1 30); do
  pgrep -f "$PATTERN" >/dev/null && break
  sleep 2
done

while pgrep -f "$PATTERN" >/dev/null; do
  ts=$(grep -E "total_timesteps" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | tail -1)
  rew=$(grep -E "ep_rew_mean" "$LOG" 2>/dev/null | tail -1 | grep -oE "[-0-9.]+" | tail -1)
  len=$(grep -E "ep_len_mean" "$LOG" 2>/dev/null | tail -1 | grep -oE "[-0-9.]+" | tail -1)
  ram=$(free -m | awk '/^Mem:/{print $7}')
  now=$(date '+%Y-%m-%d %H:%M:%S')
  pct=$(awk "BEGIN{printf \"%.1f\", ${ts:-0}/$TOTAL*100}")
  echo "| $now | ${ts:-?} | ${pct}% | ${rew:-?} | ${len:-?} | ${ram} MiB |" >> "$MD"
  if [ -n "$rew" ] && [ "$(awk "BEGIN{print ($rew > $best)?1:0}")" = "1" ]; then
    best="$rew"
    latest=$(ls -t "$CKPT"/sac_donkey_*_steps.zip 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
      cp -f "$latest" "$BEST/best.zip"
      echo "| ^^ 🏔 new peak ep_rew_mean=$rew at step ${ts:-?} — saved $(basename "$latest") -> best/best.zip |||||" >> "$MD"
    fi
  fi
  sleep "$INTERVAL"
done

echo "" >> "$MD"
echo "**Run '$RUN' ended $(date '+%Y-%m-%d %H:%M:%S')** — best ep_rew_mean=$best (model at \`checkpoints/$RUN/best/best.zip\`)." >> "$MD"
