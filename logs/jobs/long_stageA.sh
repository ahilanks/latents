#!/bin/bash
# GPU 0 (2026-11-12 09:05 UTC): is the precision of a whole-step thought limited by training time? Thought stage A on
# the cuts-held version, 3x longer (9000 x 256), then the evals. One change vs R4n6-3B-s0-think-fix (3000 steps):
# own thoughts 11-12% (units digit 85%, tens digit 13%), input-head loss still falling at the end.
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=0
A=R4n6-3B-s0-think-fix-long
python scripts/think_train.py --init runs/R4n6-3B-s0 --name=$A --seed 0 --steps 9000 --batch 256 --teacher_only \
  --dec_true 1.0 --lat reg=none --lat loss_weight=len --lat in_head=true --lat cut_from=runs/R4n6-3B-s0 \
  --save_every 3000 --log_every 50 > logs/think/$A.log 2>&1 || { echo "LONG_A_FAILED"; exit 1; }
python scripts/think_withhold.py runs/$A > logs/think/withhold-$A.log 2>&1 || echo "EVAL_FAILED"
python scripts/hf_upload_runs.py $A >> logs/think/$A.log 2>&1 || echo "HF_UPLOAD_FAILED"
echo "[$(date -u +%H:%M)] LONG_A_DONE"
