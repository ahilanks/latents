#!/bin/bash
# GPU 1 (2026-11-12, after the truncated bridge was stopped at step 400: all-own 44% -> 2.4%, anchors faded to 0 each stage).
# One change vs coconut-anchor-tb1-l2: the anchors stay at full weight (--anchor_fade none). The thoughts are trained on the
# model's own rollouts with their latent targets + the latent loss on what follows (one-hop backprop). Tests whether own-
# rollout training + the signal from what follows improves on teacher-forced think-base-l2 (all-own 61% on 2-6 steps).
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=1
python scripts/think_train.py --init runs/think-base-l2 --name=coconut-anchor-const-l2 --mode coconut --steps 4000 --batch 256 \
  --lr 2e-4 --max_stage 7 --stage_steps 400 --p_mix 0.3 --c 0 --beta_anchor 1.0 --anchor_fade none --bptt 1 --dec_true 1.0 \
  --save_every 400 --log_every 25 > logs/think/coconut-anchor-const-l2.log 2>&1 || { echo "COCONUT_CONST_FAILED"; exit 1; }
echo "[$(date -u +%H:%M)] trained; evals"
for s in 400 800 1200 1600 2000 2400 2800 3200 3600; do
  python scripts/think_withhold.py runs/coconut-anchor-const-l2-s$s --only m=0,m=all --tag stage >> logs/think/coconut-anchor-const-l2-stages.log 2>&1
done
python scripts/think_withhold.py runs/coconut-anchor-const-l2 > logs/think/withhold-coconut-anchor-const-l2.log 2>&1
python scripts/think_decode.py runs/coconut-anchor-const-l2 > logs/think/decode-coconut-anchor-const-l2.log 2>&1
python scripts/think_withhold.py runs/coconut-anchor-const-l2 --data_dir data/task_arith_long --tag long > logs/think/withhold-coconut-anchor-const-l2-long.log 2>&1
python scripts/hf_upload_runs.py coconut-anchor-const-l2 >> logs/think/coconut-anchor-const-l2.log 2>&1 || echo "HF_UPLOAD_FAILED"
echo "[$(date -u +%H:%M)] COCONUT_CONST_DONE"
