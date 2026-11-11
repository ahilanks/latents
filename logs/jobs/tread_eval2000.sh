#!/bin/bash
# n=2000 side-by-side evals (all 2000 test problems, 2-8 steps) on GPU 1
cd /home/ubuntu/latents && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
python scripts/latent_cot.py runs/T-read --k teacher,0 --n 2000 --n_cot 0 --name=T-read-2000
python scripts/latent_cot.py runs/C-B1-plain-cot runs/T-base runs/C-L1-latent-cot --k 0 --n 2000 --n_cot 2000 --name=cot-2000
echo EVAL2000_DONE
