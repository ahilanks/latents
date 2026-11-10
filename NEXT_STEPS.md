# Next steps

Written Nov 10, 2026, after the 3B runs. Two parts: first check that the current result holds up, then build on it.

## Where things stand

The headline is 1.107 BPB at 3B bytes, vs 1.275 for plain H-Net and 1.082 for BPE. Most of that drop isn't the latent loss:

| @3B bytes | val BPB |
|---|---|
| H-Net N6·T9 | 1.275 |
| H-Net N5·T7, no latent loss (1 seed) | 1.136 |
| + latent loss (2 seeds) | 1.107 |
| BPE | 1.082 |

- **The latent loss adds −0.030 BPB (~2.6%).** That's against a single seed of the control.
- **It shrank from −0.089 at 0.6B to −0.030 at 3B.** A straight line in log-data through those two points hits zero around 6–7B bytes. Two points don't make a trend, but it's the main risk.
- **It costs extra compute.** The EMA target encoder adds ~5.4 MFLOP/byte (~6%), and the router makes slightly more chunks than the control.
- **Effective compute multiplier: about 1.15–1.4×.** The control's curve falls $0.226 / \log_2 5 \approx 0.097$ BPB per doubling of data. So 0.030 BPB is worth $2^{0.31} \approx 1.24\times$ compute, or about 1.16× net of the extra 7%. The top of the range assumes the curve flattens after 3B.
- **The N6·T9 baseline looks under-tuned.** It loses to a 9M-param byte Transformer. An outside reproduction ([bigwolfeman/hnet-research](https://github.com/bigwolfeman/hnet-research), unverified) has 1-stage H-Net ~0.05 behind BPE at a similar scale, which matches our N5·T7 control. So "closes 87% of the gap to BPE" mostly measures a weak baseline.

**Rough rating: 4/10 for now.** The setup is new as far as a literature search found: a JEPA-style next-latent loss on *learned* dynamic chunks, plus the observation that this loss can push chunk boundaries onto word edges. The BPB gain is real but small and shrinking. It goes to 6–7 if the gain holds as a steady compute multiplier at 10B bytes and also shows up out of distribution.

## Part 1: check the current result

Commands assume the job-list format of `scripts/run_queue.sh`. Times are for one A100.

**1. Second seed of the no-latent control at 3B (~1.5 h).** This firms up the −0.030.
```
python -m latents.train configs/s0_hnet.yaml group=Final-3B name=f3_plain_N5_T7_s1 model.N=5 model.arch_layout=[m4,[T7],m4] train.lr=2e-3 train.seed=1
```

**2. Compute-matched control (~1.6 h).** Give the control the ~7% extra compute the latent run gets (EMA encoder plus extra chunks). Or plot BPB against FLOPs with the EMA forward counted; it's currently left out of `train_flops`.
```
python -m latents.train configs/s0_hnet.yaml group=Final-3B name=f3_plain_N5_T7_3p2B model.N=5 model.arch_layout=[m4,[T7],m4] train.lr=2e-3 train.total_bytes=3.2e9
```

**3. 10B bytes, latent vs no-latent (~5 h each).** This is the decisive one: does the gain survive more data? The train split has 10.42B bytes.
```
python -m latents.train configs/hc_best.yaml group=Scale-10B name=f10_D3_mlp_N5_T7_s0 model.N=5 model.arch_layout=[m4,[T7],m4] train.total_bytes=1e10 train.eval_every=4000
python -m latents.train configs/s0_hnet.yaml group=Scale-10B name=f10_plain_N5_T7_s0 model.N=5 model.arch_layout=[m4,[T7],m4] train.lr=2e-3 train.total_bytes=1e10 train.eval_every=4000
```
Compare the curves at matching steps, not just the final number. If the gap keeps closing, the latent loss is mostly an early-training speedup, the same pattern as "REPA Works Until It Doesn't" (arXiv 2505.16792).

**4. Turn the latent weight down late in training (needs a small code change).** Add a `beta_tf` schedule to `latents/latent.py`, e.g. constant, then decayed to 0 during the WSD cooldown. If the loss mainly helps early, this could keep the gain without the late cost. Try it at 0.6B first, 2 seeds.

**5. Eval-only benchmarks on the existing checkpoints (no training).**
- **Out-of-distribution BPB:**
  - code
  - Chinese: there are no spaces, so this tests whether word-aligned chunks are just a whitespace trick
  - a few FineWeb-2 languages
  - Wikipedia
- **Robustness:** BPB on perturbed text, and perturbed HellaSwag with the BLT/H-Net perturbations (AntSpeak, Drop, RandomCase, Repeat, UpperCase).
- **Zero-shot tasks that score above chance at ~30M params:** BLiMP, LAMBADA, ARC-Easy, SciQ, PIQA, HellaSwag. Score them by BPB of the answer continuation; at this size that's less noisy than accuracy (Heineman et al., arXiv 2508.13144).
- **Skip:** MMLU, WinoGrande, ARC-Challenge, OpenBookQA and CUTE. They sit at chance at this size.
- **Chunk quality:**
  - boundary F1 against gold segmentations: SIGMORPHON morphemes, and Chinese word segmentation (PKU/MSR)
  - linear probes on the chunk latents
  - k-step rollout error

Needs: a byte-level adapter for lm-eval-harness, and a BPB script that takes any text file.

## Part 2: build on it

### 6. Multilevel: 2-stage H-Net with a latent loss at each level

Bytes group into chunks, and chunks group into bigger chunks (phrases, maybe sentences). Each level predicts its own next latent. This is close to the hierarchical JEPA LeCun proposed in 2022. HiLP (arXiv 2608.05806) does something similar on tokens, not learned chunks.

Why it's worth trying:
- 2-stage beats 1-stage in the H-Net paper, and reportedly beats BPE at small scale too (bigwolfeman: 1.09 vs 1.11 BPB at 42M params, unverified).
- It sets up item 7: one top-level latent step would cover something like 10–25 bytes instead of ~3.5.

Steps:
1. Compute-matched 2-stage baseline at 3B, no latent loss.
2. Latent loss at stage 1 only, stage 2 only, and both, with paired seeds.
3. Boundary stats at stage 2: do the big chunks line up with phrases or sentences?

Risks:
- Two routers to keep stable. Our N=5 router already drifted from 0.20 to 0.29 chunks per byte at 3B.
- More hyperparameters and more seed variance.

### 7. Latent chain of thought: think without decoding

While the model is "thinking," it doesn't need to spell out bytes. It can feed its predicted next-chunk latent $\hat c_{j+1}$ back into the main net, and only decode when it writes the answer.

Savings per chunk, using the final model (~3.45 bytes per chunk):

$$
\text{via bytes} \approx \underbrace{62}_{\text{main net}} + \underbrace{3.45 \times 11.4}_{\text{enc + dec per byte}} \approx 101 \text{ MFLOP}, \qquad \text{latent step} \approx 62 \text{ MFLOP}
$$

- About 1.6× fewer FLOPs per step.
- One sequential step instead of ~3.45, since the byte decoder generates one byte at a time.
- At larger scale the FLOP saving shrinks, because the main net dominates compute. The saving in sequential steps stays, and it grows with item 6's bigger chunks.

Why this method fits:
- The latents are trained to match real encoder states, so a latent "thought" should in principle be decodable and checkable.
- Coconut (Hao et al., 2024) feeds back raw hidden states that aren't trained to match anything.
- We already have both pieces:
  - rollout (S2, `latents/rollout.py`)
  - the decoder reading predicted latents (S3, `latent.decode_from_pred`)

Steps:
1. Train with k-step rollout (k = 2, 4, 8) using predictor-only gradients, and measure how fast rollout error grows.
2. On a synthetic multi-step task (arithmetic, or graph reachability like Coconut's ProsQA), compare three setups:
   - no chain of thought
   - byte-level chain of thought
   - k latent steps

   Measure accuracy against FLOPs and against sequential steps. A 33M model can't do chain of thought on natural language, so stay synthetic until the model is much bigger.
3. Decode the intermediate latents to see whether the thoughts are readable.

Risks:
- S2 rollouts were weak and varied by seed (best rollout cosine 0.73), and errors compound over steps.
- Making the decoder read predicted latents cost ~0.03 BPB at 0.6B (S3d vs S2d).

If item 3 shows the BPB gain disappearing, this becomes the main reason the latent loss matters. It would buy a usable latent space, which plain H-Net doesn't have.

## Order

1 → 2 → 3 first: cheap, and everything else depends on them. Run 5 alongside them, since it needs no training. Then 4, then 6, then 7. Multilevel comes before latent reasoning because it's a direct BPB experiment, and it makes the latent steps big enough for reasoning to pay off.
