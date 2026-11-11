#!/bin/bash
# L2 vs L1 from scratch, same budget (user 2026-11-11):
#   1. pretrain M1 with the L2 latent loss: identical to L-M1-s0 (r6_detach_trunk, 3e9 bytes, seed 0) except latent.dist=l2
#   2. the same thought stage on both bases (L-M1-s0 = L1, L-M1-l2-s0 = L2), current code (padding fix), then evals
cd /home/ubuntu/latents && source .venv/bin/activate && set -a && source .env && set +a
echo "[$(date -u +%H:%M)] waiting for T-read2 to free GPU0"
while pgrep -f "name=T-read2 " >/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M)] pretrain L-M1-l2-s0 on GPU0"
CUDA_VISIBLE_DEVICES=0 python -m latents.train configs/ladder/r6_detach_trunk.yaml name=L-M1-l2-s0 train.seed=0 group=ladder-P1 \
  train.total_bytes=3e9 train.eval_every=2000 latent.dist=l2 > logs/think/L-M1-l2-s0.log 2>&1 || { echo "PRETRAIN_FAILED"; exit 1; }
echo "[$(date -u +%H:%M)] pretrain done: $(python -c "import json; print(json.load(open('runs/L-M1-l2-s0/summary.json'))['val/bpb'])") bpb (L1 base: 1.2989)"
stage() {  # $1 gpu, $2 base, $3 name
  CUDA_VISIBLE_DEVICES=$1 python scripts/think_train.py --init runs/$2 --name=$3 --steps 3000 --batch 256 --teacher_only --dec_true 1.0 --log_every 50 \
  && CUDA_VISIBLE_DEVICES=$1 python scripts/think_diag.py runs/$3 --n 2000 \
  && CUDA_VISIBLE_DEVICES=$1 python scripts/latent_cot.py runs/$3 --k teacher,0 --n 2000 --n_cot 0 --name=$3-2000 \
  && CUDA_VISIBLE_DEVICES=$1 python scripts/think_probe.py runs/$3
}
stage 0 L-M1-l2-s0 think-base-l2 > logs/think/think-base-l2.log 2>&1 &
stage 1 L-M1-s0 think-base-l1 > logs/think/think-base-l1.log 2>&1 &
wait
echo "[$(date -u +%H:%M)] L2_SCRATCH_DONE"
