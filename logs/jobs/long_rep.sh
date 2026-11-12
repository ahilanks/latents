#!/bin/bash
# Replicates of the 9000-step cuts-held thought stage (2026-11-12 ~11:40 UTC). R4n6-3B-s0-think-fix-long solved 98.5% of
# whole chains from its own thoughts (12% at 3000 steps); independent check 98.7%. One run per GPU, then the pod stops.
# usage: bash logs/jobs/long_rep.sh <gpu> <pretrained run> <think seed> <name>
GPU=$1; BASE=$2; SEED=$3; A=$4
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=$GPU
echo "[$(date -u +%H:%M)] $A: 9000-step thought stage, cuts held (pretrain $BASE, think seed $SEED)"
python scripts/think_train.py --init runs/$BASE --name=$A --seed $SEED --steps 9000 --batch 256 --teacher_only \
  --dec_true 1.0 --lat reg=none --lat loss_weight=len --lat in_head=true --lat cut_from=runs/$BASE \
  --save_every 3000 --log_every 50 > logs/think/$A.log 2>&1 || { echo "REP_FAILED $A"; exit 1; }
python scripts/think_withhold.py runs/$A > logs/think/withhold-$A.log 2>&1 || echo "EVAL_FAILED $A"
python scripts/hf_upload_runs.py $A >> logs/think/$A.log 2>&1 || echo "HF_UPLOAD_FAILED $A"
echo "[$(date -u +%H:%M)] REP_DONE $A"
