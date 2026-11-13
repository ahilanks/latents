#!/bin/bash
# (2026-11-12 ~20:10 PT) End-to-end reasoning stage (same recipe as e2e32_long.sh: second head, per-byte losses, ratio loss
# in 32-byte windows, 9000 x 256, nothing frozen) on the math-mix pretrain with the second head (NM1, the one that keeps
# word cuts on math) vs the web-only pretrain of the same size (RT4n6r), two seeds each, two runs per GPU.
# Starts after the mix-split queue (pid 17686) finishes.
cd /workspace/latents
while kill -0 17686 2>/dev/null; do sleep 60; done
bash logs/jobs/e2e32_long.sh 0 NM1-s0 0 > logs/think/e2e32_NM1-s0.out 2>&1 &
bash logs/jobs/e2e32_long.sh 0 RT4n6r-s0 0 > logs/think/e2e32_RT4n6r-s0.out 2>&1 &
bash logs/jobs/e2e32_long.sh 1 NM1-s1 1 > logs/think/e2e32_NM1-s1.out 2>&1 &
bash logs/jobs/e2e32_long.sh 1 RT4n6r-s1 1 > logs/think/e2e32_RT4n6r-s1.out 2>&1 &
wait
echo "[$(date -u +%H:%M)] E2E_NEXT_ALL_DONE"
