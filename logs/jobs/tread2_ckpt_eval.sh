#!/bin/bash
# Evaluate each T-read2 checkpoint (teacher, direct) on all 2000 test problems as it appears; GPU 1.
cd /home/ubuntu/latents && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
for r in T-read2-s1600 T-read2-s3200 T-read2-s4800 T-read2; do
  while [ ! -f runs/$r/final.pt ]; do
    pgrep -f "think_train.py.*T-read2" >/dev/null || { [ -f runs/$r/final.pt ] || { echo "CKPT_EVAL_ABORT trainer gone before $r"; exit 1; }; }
    sleep 30
  done
  sleep 20
  python scripts/latent_cot.py runs/$r --k teacher,0 --n 2000 --n_cot 0 --name=$r-2000 2>&1 | grep -E "latent_kteacher|direct|latent_k0|Traceback|Error" | sed "s/^/[$r] /"
done
echo CKPT_EVAL_DONE
