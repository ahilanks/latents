#!/bin/bash
# GPU 0 (2026-11-12 01:52): pretrain replicate of coconut-anchor-const-l2. Identical recipe, starting from the crossed model
# think-l2-p1t0 (pretrain L-M1-l2-s1, thought seed 0; all own 83.3% on 2-6 steps) instead of think-base-l2 (pretrain s0;
# 61.0%). Waits for the crossed run's chain to finish.
until grep -q 'L2_CROSSED_DONE\|CROSSED_FAILED' /workspace/latents/logs/think/l2_crossed.out 2>/dev/null; do sleep 30; done
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=0
python scripts/think_train.py --init runs/think-l2-p1t0 --name=coconut-anchor-const-p1 --mode coconut --steps 4000 --batch 256 \
  --lr 2e-4 --max_stage 7 --stage_steps 400 --p_mix 0.3 --c 0 --beta_anchor 1.0 --anchor_fade none --bptt 1 --dec_true 1.0 \
  --save_every 400 --log_every 25 > logs/think/coconut-anchor-const-p1.log 2>&1 || { echo "COCONUT_CONST_P1_FAILED"; exit 1; }
echo "[$(date -u +%H:%M)] trained; evals"
for s in 400 800 1200 1600 2000 2400 2800 3200 3600; do
  python scripts/think_withhold.py runs/coconut-anchor-const-p1-s$s --only m=0,m=all --tag stage >> logs/think/coconut-anchor-const-p1-stages.log 2>&1
done
python scripts/think_withhold.py runs/coconut-anchor-const-p1 > logs/think/withhold-coconut-anchor-const-p1.log 2>&1
python scripts/think_decode.py runs/coconut-anchor-const-p1 > logs/think/decode-coconut-anchor-const-p1.log 2>&1
python scripts/think_withhold.py runs/coconut-anchor-const-p1 --data_dir data/task_arith_long --tag long > logs/think/withhold-coconut-anchor-const-p1-long.log 2>&1
python scripts/hf_upload_runs.py coconut-anchor-const-p1 >> logs/think/coconut-anchor-const-p1.log 2>&1 || echo "HF_UPLOAD_FAILED"
echo "[$(date -u +%H:%M)] COCONUT_CONST_P1_DONE"
