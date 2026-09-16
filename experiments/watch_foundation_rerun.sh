#!/usr/bin/env bash
# Keep the foundation rerun queue alive and refresh the results report.
#
# - restarts experiments/run_foundation_rerun.sh if it has exited while no
#   training process is alive (finished runs are skipped via their .done
#   sentinels, so restarts are idempotent)
# - waits instead of restarting when only the queue script died but a training
#   process is still alive, so a seed is never trained twice on one GPU
# - regenerates the CSV + appendix report every 15 minutes
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p experiments/logs/foundation_rerun

while true; do
  if ! pgrep -f "[r]un_foundation_rerun.sh" >/dev/null; then
    if pgrep -f "[f]inetune_main.py" >/dev/null; then
      echo "[watchdog] queue script missing but a training process is alive; waiting ($(date '+%F %T'))" >> experiments/logs/foundation_rerun/watchdog.log
    else
      echo "[watchdog] restarting queue ($(date '+%F %T'))" >> experiments/logs/foundation_rerun/watchdog.log
      setsid nohup bash experiments/run_foundation_rerun.sh >> experiments/logs/foundation_rerun/queue_main.log 2>&1 < /dev/null &
    fi
  fi
  python experiments/collect_foundation_results.py >> experiments/logs/foundation_rerun/collector.log 2>&1
  sleep 900
done
