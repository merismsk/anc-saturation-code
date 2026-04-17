#!/usr/bin/env bash
# Launches the Phase 2c (data-scaled) 5-seed sweep via run_ieee_sweep_mac.sh and the monitor.
#
# Usage:
#   bash start_overnight_sweep.sh
#
# In the morning, read outputs/sweep_status.md for a full summary.
# Stop everything with:
#   bash stop_overnight_sweep.sh     # or kill $(cat outputs/sweep.pid) $(cat outputs/monitor.pid)

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p outputs
SWEEP_LOG="outputs/ieee_m4_sweep.log"
MONITOR_LOG="outputs/monitor.log"

# Refuse to launch if a previous sweep is still running
if [ -f outputs/sweep.pid ] && kill -0 "$(cat outputs/sweep.pid)" 2>/dev/null; then
  echo "ERROR: another sweep is already running (pid $(cat outputs/sweep.pid))."
  echo "       stop it first with: bash stop_overnight_sweep.sh"
  exit 1
fi

echo "== Launching overnight sweep =="
echo "Log:     $SWEEP_LOG"
echo "Monitor: $MONITOR_LOG"
echo "Status:  outputs/sweep_status.md (rewritten every ~2 min)"
echo ""

# Start the sweep in the background
export PYTHONUNBUFFERED=1
nohup caffeinate -i bash run_ieee_sweep_mac.sh > "$SWEEP_LOG" 2>&1 &
SWEEP_PID=$!
echo "$SWEEP_PID" > outputs/sweep.pid
echo "Sweep PID:   $SWEEP_PID"

# Give the sweep a moment to create the log file
sleep 2

# Start the monitor
nohup python3 monitor_sweep.py \
    --log "$SWEEP_LOG" \
    --output-root outputs \
    --expected-seeds 42 123 456 789 2024 \
    --sweep-pid "$SWEEP_PID" \
    --poll-sec 60 \
    --stall-sec 900 \
    > "$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
echo "$MONITOR_PID" > outputs/monitor.pid
echo "Monitor PID: $MONITOR_PID"

echo ""
echo "== Both running =="
echo "Check progress anytime with:"
echo "    cat outputs/sweep_status.md"
echo "    tail -f $SWEEP_LOG"
echo ""
echo "Kill everything with:"
echo "    bash stop_overnight_sweep.sh"
