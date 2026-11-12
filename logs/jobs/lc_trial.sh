#!/bin/bash
# GPU 0 (2026-11-12 04:40 UTC): first thought-stage trial on a learned-cut model (latents/think_lc.py). Base: RT4w01-s0
# (0.6B, chunk-local targets, ratio weight 0.1, word-start cuts). Teacher-forced on the traces, 1500 x 256, then the
# read / n-steps / all-own evals. An early look before the 3B runs.
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=0
python scripts/think_train.py --init runs/RT4w01-s0 --name=lc-trial-w01 --steps 1500 --batch 256 --teacher_only \
  --dec_true 1.0 --log_every 50 > logs/think/lc-trial-w01.log 2>&1 || { echo "LC_TRIAL_FAILED"; exit 1; }
python scripts/think_withhold.py runs/lc-trial-w01 > logs/think/withhold-lc-trial-w01.log 2>&1 || echo "LC_EVAL_FAILED"
echo "[$(date -u +%H:%M)] LC_TRIAL_DONE"
