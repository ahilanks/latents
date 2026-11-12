#!/bin/bash
# Practice stage only, from a finished thought-stage-A run (own thoughts replace written steps; each keeps its latent
# target), then the evals. usage: bash logs/jobs/practice_lc.sh <gpu> <stage-A run> <practice run name> <seed> [extra args]
GPU=$1; A=$2; P=$3; SEED=$4; shift 4; EXTRA="$@"
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=$GPU
L=logs/think
echo "[$(date -u +%H:%M)] $P: practice stage from $A ($EXTRA)"
python scripts/think_train.py --init runs/$A --name=$P --seed $SEED --mode coconut --steps 4000 --batch 256 --lr 2e-4 \
  --max_stage 7 --stage_steps 400 --p_mix 0.3 --c 0 --beta_anchor 1.0 --anchor_fade none --bptt 1 --dec_true 1.0 \
  --lat reg=none $EXTRA --save_every 800 --log_every 25 > $L/$P.log 2>&1 || { echo "PRACTICE_FAILED $P"; exit 1; }
python scripts/think_withhold.py runs/$P > $L/withhold-$P.log 2>&1 || echo "EVAL_FAILED $P"
python scripts/hf_upload_runs.py $A $P >> $L/$P.log 2>&1 || echo "HF_UPLOAD_FAILED $P"
echo "[$(date -u +%H:%M)] PRACTICE_DONE $P"
