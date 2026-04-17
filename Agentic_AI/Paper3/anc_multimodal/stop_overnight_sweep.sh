#!/usr/bin/env bash
# Gracefully stops the overnight sweep and monitor.
set -eu
cd "$(dirname "$0")"

for name in sweep monitor; do
  f="outputs/${name}.pid"
  if [ -f "$f" ]; then
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (pid $pid)..."
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    else
      echo "$name pid $pid not running"
    fi
    rm -f "$f"
  fi
done
echo "Done."
