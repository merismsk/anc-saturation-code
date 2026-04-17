#!/usr/bin/env bash
# Launches Phase 2c A100 sweep + monitor. Linux-adapted from Mac version:
# no caffeinate (server doesn't sleep), nohup-based background launch.
#
# Usage:
#   bash start_overnight_sweep_a100.sh
#
# In the morning, read outputs/sweep_status.md for the dashboard.
# Kill everything with:
#   bash stop_overnight_sweep.sh

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p outputs
SWEEP_LOG="outputs/ieee_a100_sweep.log"
MONITOR_LOG="outputs/monitor_a100.log"

# Refuse to launch if a previous sweep is still running
if [ -f outputs/sweep.pid ] && kill -0 "$(cat outputs/sweep.pid)" 2>/dev/null; then
  echo "ERROR: another sweep is already running (pid $(cat outputs/sweep.pid))."
  echo "       stop it first with: bash stop_overnight_sweep.sh"
  exit 1
fi

# Pre-flight: verify CUDA is visible to Python (fast check, no model load)
python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA not available. Check nvidia-smi and torch installation.')
    sys.exit(1)
print(f'CUDA: {torch.cuda.get_device_name(0)}  (devices={torch.cuda.device_count()})')
" || exit 1

# Pre-flight: verify ffmpeg available (required by pydub for FLAC)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found. Install with: sudo apt install -y ffmpeg libsndfile1"
  exit 1
fi

# Pre-flight: verify pydub importable
python3 -c "import pydub" 2>/dev/null || {
  echo "ERROR: pydub not installed. Install with: pip install pydub"
  exit 1
}

echo "== Launching A100 overnight sweep =="
echo "Log:     $SWEEP_LOG"
echo "Monitor: $MONITOR_LOG"
echo "Status:  outputs/sweep_status.md (rewritten every ~60 s)"
echo ""

# Start the sweep in the background (no caffeinate on Linux)
export PYTHONUNBUFFERED=1
nohup bash run_ieee_sweep_a100.sh > "$SWEEP_LOG" 2>&1 &
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
