#!/bin/bash
# Latent reasoning on a pretrained learned-cut model: thought stage A (teacher-forced on the traces), evals, practice
# stage (own thoughts replace written steps; each keeps its latent target), evals. --lat overrides apply to both stages.
# usage: bash logs/jobs/reason_lc.sh <gpu> <pretrained run> <tag> <seed> [extra think_train args, e.g. --lat loss_weight=len]
GPU=$1; BASE=$2; TAG=$3; SEED=$4; shift 4; EXTRA="$@"
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=$GPU
L=logs/think; A=$BASE-think$TAG; P=$BASE-practice$TAG
echo "[$(date -u +%H:%M)] $A: thought stage A ($EXTRA)"
python scripts/think_train.py --init runs/$BASE --name=$A --seed $SEED --steps 3000 --batch 256 --teacher_only \
  --dec_true 1.0 --lat reg=none $EXTRA --log_every 50 > $L/$A.log 2>&1 || { echo "THINK_FAILED $A"; exit 1; }
python scripts/think_withhold.py runs/$A > $L/withhold-$A.log 2>&1 || echo "EVAL_FAILED $A"
echo "[$(date -u +%H:%M)] $A: stage A done; practice stage"
python scripts/think_train.py --init runs/$A --name=$P --seed $SEED --mode coconut --steps 4000 --batch 256 --lr 2e-4 \
  --max_stage 7 --stage_steps 400 --p_mix 0.3 --c 0 --beta_anchor 1.0 --anchor_fade none --bptt 1 --dec_true 1.0 \
  --lat reg=none $EXTRA --save_every 800 --log_every 25 > $L/$P.log 2>&1 || { echo "PRACTICE_FAILED $P"; exit 1; }
python scripts/think_withhold.py runs/$P > $L/withhold-$P.log 2>&1 || echo "EVAL_FAILED $P"
python scripts/hf_upload_runs.py $A $P >> $L/$A.log 2>&1 || echo "HF_UPLOAD_FAILED $A"
echo "[$(date -u +%H:%M)] REASON_DONE $BASE$TAG"
