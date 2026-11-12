#!/bin/bash
# GPU 0 (2026-11-11 23:25, user: "launch it"): Coconut's curriculum in latent space (latents/think.py loss_coconut).
# Init: L-M1-l2-s0 (pretrained only; stage 0 = the written traces). Stage s replaces the first s written steps with own
# thoughts (one per replaced 4-byte chunk), trained only by the latent loss on the remaining text + answer through the
# thought chain; replaced bytes deleted from the encoded text; p_mix 0.3; optimizer reset per stage. Stages 0-7 x 400
# steps, then 800 more at stage 7 (all steps replaced). Batch 256, dec_true 1, same LR schedule as the thought stage.
cd /workspace/latents && source .venv/bin/activate && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=0
python scripts/think_train.py --init runs/L-M1-l2-s0 --name=coconut-l2 --mode coconut --steps 4000 --batch 256 \
  --max_stage 7 --stage_steps 400 --p_mix 0.3 --c 0 --dec_true 1.0 --save_every 400 --log_every 25 \
  > logs/think/coconut-l2.log 2>&1 || { echo "COCONUT_FAILED"; exit 1; }
echo "[$(date -u +%H:%M)] trained; evals"
# accuracy with all thoughts own (fixed count = the reasoning's chunk count, Coconut's constant-length halting) after each stage
for s in 400 800 1200 1600 2000 2400 2800 3200 3600; do
  python scripts/think_withhold.py runs/coconut-l2-s$s --only m=0,m=all --tag stage >> logs/think/coconut-l2-stages.log 2>&1
done
python scripts/think_withhold.py runs/coconut-l2 > logs/think/withhold-coconut-l2.log 2>&1
python scripts/think_decode.py runs/coconut-l2 > logs/think/decode-coconut-l2.log 2>&1
python scripts/think_withhold.py runs/coconut-l2 --data_dir data/task_arith_long --tag long > logs/think/withhold-coconut-l2-long.log 2>&1
python scripts/hf_upload_runs.py coconut-l2 >> logs/think/coconut-l2.log 2>&1 || echo "HF_UPLOAD_FAILED"
echo "[$(date -u +%H:%M)] COCONUT_L2_DONE"
