#!/bin/bash
# GPU 0, after coconut-l2's chain (user: "run it"): the crossed-seed run. Thought seed 0 (as think-base-l2) on the seed-1
# pretrain (L-M1-l2-s1). Separates pretrain-seed from thought-stage-seed variance: think-base-l2 (s0/s0) read 97.2% and
# computed 61% whole chains; think-base-l2-s1 (s1/s1) 81.1% and 78%.
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=0
until grep -q "COCONUT_L2_DONE\|COCONUT_FAILED" logs/think/coconut_l2.out 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M)] crossed run think-l2-p1t0"
python scripts/think_train.py --init runs/L-M1-l2-s1 --name=think-l2-p1t0 --seed 0 --steps 3000 --batch 256 --teacher_only \
  --dec_true 1.0 --log_every 50 > logs/think/think-l2-p1t0.log 2>&1 || { echo "CROSSED_FAILED"; exit 1; }
python scripts/think_diag.py runs/think-l2-p1t0 --n 2000 >> logs/think/think-l2-p1t0.log 2>&1
python scripts/latent_cot.py runs/think-l2-p1t0 --k teacher,0 --n 2000 --n_cot 0 --name=think-l2-p1t0-2000 >> logs/think/think-l2-p1t0.log 2>&1
python scripts/think_withhold.py runs/think-l2-p1t0 > logs/think/withhold-think-l2-p1t0.log 2>&1
python scripts/hf_upload_runs.py think-l2-p1t0 >> logs/think/think-l2-p1t0.log 2>&1 || echo "HF_UPLOAD_FAILED"
echo "[$(date -u +%H:%M)] L2_CROSSED_DONE"
