#!/bin/bash
# Spaced-format test (user, 2026-11-12 ~09:55 UTC): the same problems written with a space between every piece
# ("b = 17 + 25 = 42 ."), so the pretrained word-start cuts give ~7 thoughts per written step instead of 1 (data in
# data/task_arith_sp, same 2000 test problems). Thought stage A + evals on the same pretrained model, nothing else changed.
# usage: bash logs/jobs/spaced_test.sh <gpu> <tag> <extra think_train args>
GPU=$1; TAG=$2; shift 2; EXTRA="$@"
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=$GPU
A=R4n6-3B-s0-think-sp$TAG
echo "[$(date -u +%H:%M)] $A: thought stage A, spaced format ($EXTRA)"
python scripts/think_train.py --init runs/R4n6-3B-s0 --name=$A --seed 0 --steps 3000 --batch 256 --teacher_only --spaced \
  --data_dir data/task_arith_sp --dec_true 1.0 --lat reg=none --lat loss_weight=len --lat in_head=true $EXTRA \
  --log_every 50 > logs/think/$A.log 2>&1 || { echo "SPACED_FAILED $A"; exit 1; }
python scripts/think_withhold.py runs/$A > logs/think/withhold-$A.log 2>&1 || echo "EVAL_FAILED $A"
python scripts/hf_upload_runs.py $A >> logs/think/$A.log 2>&1 || echo "HF_UPLOAD_FAILED $A"
echo "[$(date -u +%H:%M)] SPACED_DONE $A"
