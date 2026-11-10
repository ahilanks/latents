#!/usr/bin/env bash
# Run the whole ladder on this pod: preflight, then phases 1-6 and the eval suite, stopping at the first failed
# gate (run_jobs.py sends a Pushover notification if PUSHOVER_* is set in .env).
# usage: bash scripts/ladder/run_all.sh [--gpus 0,1,2]     (default: all visible GPUs)
# With 4 GPUs, give the side track its own GPU in a second terminal:
#        bash scripts/ladder/run_all.sh --gpus 0,1,2   and   python scripts/run_jobs.py scripts/ladder/side.txt --gpus 3
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
for p in preflight phase1 phase2 phase3 phase4 phase5 phase6 evals; do
    echo "=== $p ==="
    python scripts/run_jobs.py "scripts/ladder/$p.txt" "$@"
done
