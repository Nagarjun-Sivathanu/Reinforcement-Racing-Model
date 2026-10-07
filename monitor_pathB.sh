#!/usr/bin/env bash
# Detached logger + best-peak saver for the Path B run. Runs independently of any editor
# session (launch with setsid). Appends a status row to TRAINING_LOG.md every INTERVAL and
# copies the latest checkpoint to best/best.zip whenever ep_rew_mean hits a new all-time max.
ROOT=/home/nagu/projects/Rl_Race
LOG=$ROOT/logs/train_champ3.log
MD=$ROOT/TRAINING_LOG.md
CKPT=$ROOT/checkpoints/champ_continue3
BEST=$CKPT/best
TOTAL=1000000
INTERVAL=300
PATTERN="train_rl.py --timesteps 1000000"
mkdir -p "$BEST"
best="-1000000"

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
echo "**Run ended $(date '+%Y-%m-%d %H:%M:%S')** — best ep_rew_mean=$best (model at \`checkpoints/pathB_long/best/best.zip\`)." >> "$MD"
