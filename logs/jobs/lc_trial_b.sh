#!/bin/bash
# GPU 0 (2026-11-12 05:02 UTC): thought-stage trial on the word-cut model RT4w01-s0, second try. In the first try the
# cuts died within 100 steps (cut rate 0.16 -> 0.04, only the forced boundaries left): on the narrow arithmetic text the
# VICReg term starts at 1.22 (0.02 on web text) and its gradient into the cut scores beats the ratio loss (weight 0.1).
# One change: no VICReg term in the thought stage (--lat reg=none), as in the fixed-chunk thought stages.
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=0
python scripts/think_train.py --init runs/RT4w01-s0 --name=lc-trial-w01b --steps 1500 --batch 256 --teacher_only \
  --dec_true 1.0 --lat reg=none --log_every 50 > logs/think/lc-trial-w01b.log 2>&1 || { echo "LC_TRIAL_FAILED"; exit 1; }
python scripts/think_withhold.py runs/lc-trial-w01b > logs/think/withhold-lc-trial-w01b.log 2>&1 || echo "LC_EVAL_FAILED"
echo "[$(date -u +%H:%M)] LC_TRIAL_B_DONE"
