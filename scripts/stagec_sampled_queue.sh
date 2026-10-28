#!/bin/bash
# Run one option-B simulation detached through the shared slot lock.
# usage: scripts/stagec_sampled_queue.sh <log name> <main.py args...>
cd "$(dirname "$0")/.." || exit 1
log=logs/$1.log; shift
mkdir -p logs
nohup setsid scripts/with_sim_slot.sh uv run python main.py --read-only "$@" > "$log" 2>&1 < /dev/null &
