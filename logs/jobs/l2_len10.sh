#!/bin/bash
# GPU 1 (2026-11-11 22:40 UTC, step 2 of the roadmap, approved): length generalization. One change vs think-base-l2:
# training problems have 2-10 steps (was 2-6). Evaluated on the new 2-12-step test set (data/task_arith_long), plus the
# standard 2-8 set. The base model (think-base-l2) gets the same 2-12 evals for comparison.
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=1
( python scripts/think_withhold.py runs/think-base-l2 --data_dir data/task_arith_long --tag long > logs/think/withhold-think-base-l2-long.log 2>&1
  python scripts/think_decode.py runs/think-base-l2 --data_dir data/task_arith_long --tag long > logs/think/decode-think-base-l2-long.log 2>&1 ) &
python scripts/think_train.py --init runs/L-M1-l2-s0 --name=think-l2-len10 --steps 3000 --batch 256 --teacher_only \
  --dec_true 1.0 --train_steps 2,10 --log_every 50 > logs/think/think-l2-len10.log 2>&1 || { echo "THINK_FAILED"; exit 1; }
python scripts/think_withhold.py runs/think-l2-len10 --data_dir data/task_arith_long --tag long > logs/think/withhold-think-l2-len10-long.log 2>&1
python scripts/think_decode.py runs/think-l2-len10 --data_dir data/task_arith_long --tag long > logs/think/decode-think-l2-len10-long.log 2>&1
python scripts/think_withhold.py runs/think-l2-len10 > logs/think/withhold-think-l2-len10.log 2>&1
python scripts/hf_upload_runs.py think-l2-len10 >> logs/think/think-l2-len10.log 2>&1 || echo "HF_UPLOAD_FAILED"
wait
echo "[$(date -u +%H:%M)] L2_LEN10_DONE"
