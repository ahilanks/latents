#!/bin/bash
# L2 second seed, from scratch (user 2026-11-11 21:40 UTC), on GPU 0:
#   1. pretrain L-M1-l2-s1: identical to L-M1-l2-s0 except train.seed=1 (paired L1 reference: L-M1-s1 = 1.2900)
#   2. the same thought stage as think-base-l2 (teacher-only, dec_true 1, 3000 x 256), think seed 1 -> think-base-l2-s1
#   3. the standard evals + the withhold-m read-then-compute curve (scripts/think_withhold.py)
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=0
echo "[$(date -u +%H:%M)] pretrain L-M1-l2-s1"
python -m latents.train configs/ladder/r6_detach_trunk.yaml name=L-M1-l2-s1 train.seed=1 group=ladder-P1 \
  train.total_bytes=3e9 train.eval_every=2000 latent.dist=l2 > logs/think/L-M1-l2-s1.log 2>&1 || { echo "PRETRAIN_FAILED"; exit 1; }
echo "[$(date -u +%H:%M)] pretrain done: $(python -c "import json; print(json.load(open('runs/L-M1-l2-s1/summary.json'))['val/bpb'])") bpb (L2 s0 1.2554; L1 s1 1.2900)"
python scripts/hf_upload_runs.py L-M1-l2-s1 >> logs/think/L-M1-l2-s1.log 2>&1 || echo "HF_UPLOAD_FAILED L-M1-l2-s1"
echo "[$(date -u +%H:%M)] thought stage think-base-l2-s1"
python scripts/think_train.py --init runs/L-M1-l2-s1 --name=think-base-l2-s1 --seed 1 --steps 3000 --batch 256 --teacher_only \
  --dec_true 1.0 --log_every 50 > logs/think/think-base-l2-s1.log 2>&1 || { echo "THINK_FAILED"; exit 1; }
echo "[$(date -u +%H:%M)] evals"
python scripts/think_diag.py runs/think-base-l2-s1 --n 2000 >> logs/think/think-base-l2-s1.log 2>&1
python scripts/latent_cot.py runs/think-base-l2-s1 --k teacher,0 --n 2000 --n_cot 0 --name=think-base-l2-s1-2000 >> logs/think/think-base-l2-s1.log 2>&1
python scripts/think_probe.py runs/think-base-l2-s1 >> logs/think/think-base-l2-s1.log 2>&1
python scripts/think_withhold.py runs/think-base-l2-s1 >> logs/think/withhold-think-base-l2-s1.log 2>&1 || echo "WITHHOLD_FAILED"
python scripts/hf_upload_runs.py think-base-l2-s1 >> logs/think/think-base-l2-s1.log 2>&1 || echo "HF_UPLOAD_FAILED think-base-l2-s1"
echo "[$(date -u +%H:%M)] L2_SEED1_DONE"
