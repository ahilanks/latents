#!/bin/bash
# GPU 1 (2026-11-12 ~00:30, user: "run it"): the A->B bridge. Coconut's curriculum (as coconut-l2) but starting from the
# anchored thought model think-base-l2, and the replaced thoughts keep their latent targets with a weight fading 1 -> 0
# over each stage (--beta_anchor 1), so the signal from what follows takes over. LR 2e-4 (continuing a fine-tuned model).
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=1
python scripts/think_train.py --init runs/think-base-l2 --name=coconut-anchor-l2 --mode coconut --steps 4000 --batch 256 \
  --lr 2e-4 --max_stage 7 --stage_steps 400 --p_mix 0.3 --c 0 --beta_anchor 1.0 --dec_true 1.0 --save_every 400 \
  --log_every 25 > logs/think/coconut-anchor-l2.log 2>&1 || { echo "COCONUT_ANCHOR_FAILED"; exit 1; }
echo "[$(date -u +%H:%M)] trained; evals"
for s in 400 800 1200 1600 2000 2400 2800 3200 3600; do
  python scripts/think_withhold.py runs/coconut-anchor-l2-s$s --only m=0,m=all --tag stage >> logs/think/coconut-anchor-l2-stages.log 2>&1
done
python scripts/think_withhold.py runs/coconut-anchor-l2 > logs/think/withhold-coconut-anchor-l2.log 2>&1
python scripts/think_decode.py runs/coconut-anchor-l2 > logs/think/decode-coconut-anchor-l2.log 2>&1
python scripts/think_withhold.py runs/coconut-anchor-l2 --data_dir data/task_arith_long --tag long > logs/think/withhold-coconut-anchor-l2-long.log 2>&1
python scripts/hf_upload_runs.py coconut-anchor-l2 >> logs/think/coconut-anchor-l2.log 2>&1 || echo "HF_UPLOAD_FAILED"
echo "[$(date -u +%H:%M)] COCONUT_ANCHOR_L2_DONE"
