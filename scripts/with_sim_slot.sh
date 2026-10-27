#!/bin/bash
# Run a command once one of N simulation slots is free (default 2), so parallel
# runs from several worktrees do not exhaust memory (a city run needs ~3.5 GB).
# usage: scripts/with_sim_slot.sh uv run python main.py ...
SLOTS=${SIM_SLOTS:-2}
DIR=${SIM_SLOT_DIR:-/tmp/ride-sharing-sim-slots}
mkdir -p "$DIR"
while true; do
  for i in $(seq 1 "$SLOTS"); do
    exec {fd}>"$DIR/slot$i"
    if flock -n "$fd"; then
      "$@"
      status=$?
      flock -u "$fd"
      exit $status
    fi
    exec {fd}>&-
  done
  sleep 20
done
