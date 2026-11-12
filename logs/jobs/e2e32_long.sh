#!/bin/bash
# END-TO-END reasoning stage at the length that worked with held cuts (2026-11-12 ~11:50 UTC). Nothing frozen or blocked:
# second head, per-byte losses, ratio loss in 32-byte windows, 9000 x 256. scripts/cut_drift.py on the earlier runs:
# plain end to end merges the whole reasoning; 64-byte windows merge the early steps (3 of 4 merged by step 9000);
# 32-byte windows keep ~10 cuts, now on each step's final "." instead of the next step's first letter, but that run had
# only 3000 steps (computing needed >3000 steps even with held cuts).
# usage: bash logs/jobs/e2e32_long.sh <gpu> <pretrained run> <seed>
GPU=$1; BASE=$2; SEED=$3
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=$GPU
A=$BASE-think-e2e32-long
echo "[$(date -u +%H:%M)] $A: end-to-end, 32-byte windowed ratio loss, 9000 steps"
python scripts/think_train.py --init runs/$BASE --name=$A --seed $SEED --steps 9000 --batch 256 --teacher_only \
  --dec_true 1.0 --lat reg=none --lat loss_weight=len --lat in_head=true --lat ratio_window=32 \
  --save_every 3000 --log_every 50 > logs/think/$A.log 2>&1 || { echo "E2E32_FAILED $A"; exit 1; }
python scripts/think_withhold.py runs/$A > logs/think/withhold-$A.log 2>&1 || echo "EVAL_FAILED $A"
python scripts/cut_drift.py runs/$BASE runs/$A-s3000 runs/$A-s6000 runs/$A > logs/think/drift-$A.log 2>&1 || echo "DRIFT_FAILED $A"
python scripts/hf_upload_runs.py $A >> logs/think/$A.log 2>&1 || echo "HF_UPLOAD_FAILED $A"
echo "[$(date -u +%H:%M)] E2E32_DONE $A"
