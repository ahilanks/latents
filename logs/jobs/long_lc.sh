#!/bin/bash
# Long run of a learned-cut variant (user plan, 2026-11-12): pretrain from scratch on 3B bytes, then latent reasoning:
# thought stage A (teacher-forced on the traces), practice stage (own thoughts replace written steps, each still pulled
# toward the chunk it replaces), evals after each. The VICReg term is off in the reasoning stages (--lat reg=none): on
# the narrow arithmetic text it kills the cuts.
# usage: bash logs/jobs/long_lc.sh <gpu> <config name in configs/router> <run name> [seed]
GPU=$1; CFG=$2; NAME=$3; SEED=${4:-0}
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=$GPU
L=logs/think
echo "[$(date -u +%H:%M)] $NAME: pretrain 3B ($CFG, seed $SEED)"
python -m latents.train configs/router/$CFG.yaml name=$NAME train.seed=$SEED group=router-3B train.total_bytes=3e9 \
  train.eval_every=1000 > $L/$NAME.log 2>&1 || { echo "PRETRAIN_FAILED $NAME"; exit 1; }
echo "[$(date -u +%H:%M)] $NAME: pretrain done, bpb $(python -c "import json; print(round(json.load(open('runs/$NAME/summary.json'))['val/bpb'], 4))")"
python scripts/tokenize_compare.py runs/$NAME/cuts.json "Tokenizer=bpe" "$NAME=runs/$NAME" > $L/$NAME-cuts.log 2>&1
python scripts/eval_suite.py run runs/$NAME --tasks chunks >> $L/$NAME-cuts.log 2>&1
echo "[$(date -u +%H:%M)] $NAME: thought stage A"
python scripts/think_train.py --init runs/$NAME --name=$NAME-think --seed $SEED --steps 3000 --batch 256 --teacher_only \
  --dec_true 1.0 --lat reg=none --log_every 50 > $L/$NAME-think.log 2>&1 || { echo "THINK_FAILED $NAME"; exit 1; }
python scripts/think_withhold.py runs/$NAME-think > $L/withhold-$NAME-think.log 2>&1 || echo "EVAL_FAILED $NAME-think"
echo "[$(date -u +%H:%M)] $NAME: stage A done; practice stage"
python scripts/think_train.py --init runs/$NAME-think --name=$NAME-practice --seed $SEED --mode coconut --steps 4000 \
  --batch 256 --lr 2e-4 --max_stage 7 --stage_steps 400 --p_mix 0.3 --c 0 --beta_anchor 1.0 --anchor_fade none --bptt 1 \
  --dec_true 1.0 --lat reg=none --save_every 800 --log_every 25 > $L/$NAME-practice.log 2>&1 || { echo "PRACTICE_FAILED $NAME"; exit 1; }
python scripts/think_withhold.py runs/$NAME-practice > $L/withhold-$NAME-practice.log 2>&1 || echo "EVAL_FAILED $NAME-practice"
python scripts/hf_upload_runs.py $NAME $NAME-think $NAME-practice >> $L/$NAME.log 2>&1 || echo "HF_UPLOAD_FAILED $NAME"
echo "[$(date -u +%H:%M)] LONG_DONE $NAME"
