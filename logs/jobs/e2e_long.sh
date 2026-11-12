#!/bin/bash
# GPU 1 (2026-11-12 ~09:35 UTC): END-TO-END long thought stage on seed 1 (nothing frozen or blocked): second head,
# per-byte losses, windowed ratio loss with a wider window (64 bytes; 32 kept the thoughts but slowed reading), 9000 x 256.
# Waits for the cuts-held seed-1 stage-A eval, then replaces that chain (its practice stage is not run).
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
until grep -q "^wrote\|Traceback" logs/think/withhold-R4n6-3B-s1-think-fix.log 2>/dev/null; do sleep 15; done
sleep 5
for p in $(ps -eo pid,args | grep -E "reason_lc[.]sh 1 |name=R4n6-3B-s1-practice-fix" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 3
export CUDA_VISIBLE_DEVICES=1
A=R4n6-3B-s1-think-e2e-long
echo "[$(date -u +%H:%M)] $A: end-to-end long thought stage"
python scripts/think_train.py --init runs/R4n6-3B-s1 --name=$A --seed 1 --steps 9000 --batch 256 --teacher_only \
  --dec_true 1.0 --lat reg=none --lat loss_weight=len --lat in_head=true --lat ratio_window=64 \
  --save_every 3000 --log_every 50 > logs/think/$A.log 2>&1 || { echo "E2E_LONG_FAILED"; exit 1; }
python scripts/think_withhold.py runs/$A > logs/think/withhold-$A.log 2>&1 || echo "EVAL_FAILED"
python scripts/hf_upload_runs.py R4n6-3B-s1-think-fix $A >> logs/think/$A.log 2>&1 || echo "HF_UPLOAD_FAILED"
echo "[$(date -u +%H:%M)] E2E_LONG_DONE"
