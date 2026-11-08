#!/usr/bin/env bash
# Run a file of training commands (one per line) with P concurrent jobs on the single GPU.
# usage: scripts/run_queue.sh <cmds.txt> [P=3]
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="${2:-3}"
mkdir -p logs
grep -v '^\s*#' "$1" | grep -v '^\s*$' | xargs -P "$P" -I{} bash -c '
  name=$(echo "{}" | grep -oP "name=\K\S+" || echo job$$)
  echo "[$(date +%H:%M:%S)] start $name"
  {} > logs/$name.log 2>&1 && echo "[$(date +%H:%M:%S)] done  $name: $(grep -oP "\"val/bpb\": \K[0-9.]+" logs/$name.log | tail -1)" || echo "[$(date +%H:%M:%S)] FAIL  $name (see logs/$name.log)"
'
