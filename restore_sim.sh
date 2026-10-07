#!/usr/bin/env bash
# Restore DonkeySim's executable if Windows Defender quarantined it again.
# Root cause: Defender heuristically quarantines the Unity-built donkey_sim.exe. The real fix
# is a Defender folder exclusion (see notes at bottom); this script is the quick recovery.
#
# Usage:  ./restore_sim.sh        (run before starting the sim if it "disappeared")
set -u
DST="/mnt/c/Users/nagun/Documents/DonkeySimWin/donkey_sim.exe"
WSL_BACKUP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.sim_backup/donkey_sim.exe"
DL_COPY="/mnt/c/Users/nagun/Downloads/DonkeySimWin/DonkeySimWin/donkey_sim.exe"

if [ -f "$DST" ]; then
  echo "[restore_sim] exe already present: $DST"
  exit 0
fi
for SRC in "$WSL_BACKUP" "$DL_COPY"; do
  if [ -f "$SRC" ]; then
    cp -f "$SRC" "$DST" && echo "[restore_sim] restored exe from $SRC" && exit 0
  fi
done
echo "[restore_sim] ERROR: no source copy found (.sim_backup or Downloads). Re-download DonkeySim." >&2
exit 1

# ---------------------------------------------------------------------------
# PERMANENT FIX (do once, in an *admin* PowerShell on Windows):
#   Add-MpPreference -ExclusionPath "C:\Users\nagun\Documents\DonkeySimWin"
#   Add-MpPreference -ExclusionProcess "donkey_sim.exe"
# Or via UI: Windows Security -> Virus & threat protection -> Manage settings ->
#   Exclusions -> Add -> Folder -> C:\Users\nagun\Documents\DonkeySimWin
# ---------------------------------------------------------------------------
