# Logistics: what was done, when, and what worked

Times are UTC. ✅ worked · ❌ didn't work · ⚠️ caveat. Run names match W&B (`hnet-latents`) and the HF folders `runs/<name>`.

## 2026-11-07

**21:20 setup.** 1× A100-80GB, torch 2.8.0+cu128. Created **new** W&B project `hnet-latents` in team
entity `ahilan-uc-berkeley-electrical-engineering-computer-sciences` (the personal entity `models` is empty;
your other projects live in the team). Created **new** private HF model repo `a5ilank/hnet-latents`.

**21:25 reference choice.** Paper (Hwang, Wang & Gu 2025) only reports ≥760M-scale text results.
The best small-scale reproduction is main-horse's FLOP-matched sweep on FineWeb-10BT
(H-Net byte vs BPE at 1e16–1e18 FLOPs). Byte-vs-BPE BPB: 3e17 → 1.2125 vs 1.2060; 1e18 → 1.1272 vs 1.1130,
i.e. H-Net ≈ BPE within ~0.01 at this scale. Paper Table 2 (Large) gives the ordering:
BPE 0.756 ≈ H-Net-1s 0.755 < pool 0.780 < byte-isotropic 0.845–0.859.
⇒ **S0 target**: reproduce that ordering and those gaps on FineWeb-Edu at ~2.3e17 train FLOPs.

**21:30 ❌ install.** A plain `pip install mamba-ssm` upgraded torch to 2.15+cu13 and broke the env.
Reverted to torch 2.8.0. ✅ Fix: install prebuilt wheels with `--no-deps`
(causal-conv1d 1.5.2, mamba-ssm 2.2.5, flash-attn 2.8.3 for cu12/torch2.8/cp312). Building from source
took far too long, and mamba-ssm 2.2.6 has no torch-2.8 wheel. mamba-ssm 2.2.5 imports old `transformers`
names; ✅ an inert shim in `latents/__init__.py` covers them, so hub/tokenizers aren't downgraded. All of this is in `setup.sh`.

**21:40 data ✅.** FineWeb-Edu sample-10BT shards 000–002 → 10.42B train bytes / 2.255B GPT-2 tokens
(4.62 bytes/token; the paper uses 4.6). Val = the first 1702 docs of shard 013 (8.0MB), the same docs in both streams.
BPB for BPE uses the exact byte length of each predicted token.

**21:45 model code ✅.** Uses the official goombalab/hnet modules (Mamba2 enc/dec, routing, chunk/dechunk,
STE, residual proj, init, LR multipliers eq. 11). `forward_lm` re-implements the 1-stage forward to
expose chunks. `tests/test_equiv.py`: identical boundaries and 0.3% mean rel. logit diff vs official (bf16 noise).
⚠️ The official *packed* path asserts `max_seqlen` is an int but passes a tensor, so it's unusable as-is;
the masked path is used as the reference.

**21:50 S0 configs** (fwd MFLOP/byte ≈ 25, 3B bytes, batch 32×8192 bytes, AdamW β=(.9,.95) wd .1, WSD 10/70/20 with 1-sqrt cooldown):
BPE d512·T8 (ctx 1792 tok) · byte-isotropic d384·T5 (SWA 1024) · H-Net D=[256,512] m4–T9–m4, N=6, α=.03 ·
H-Net-pool k=6 m4–T11–m4. Speed alone: BPE .20, bytes .17, pool .30, H-Net .33 s/step.
❌ 3 concurrent jobs → 1.2 s/step each (slower in total than serial) and OOM at micro-batch 32. ✅ micro-batch 16 for H-Net, run serially.

**21:57–23:11 S0 LR sweep** (0.6B bytes, batch 262k bytes ≈ 2.3k steps; group `S0-lrsweep`), final val BPB:
H-Net lr 1e-3 / 2e-3 / 4e-3 → 1.4287 / **1.3721** / 1.3747 · BPE lr 1e-3 / 2e-3 → 1.2257 / **1.2076**.
❌ **H-Net lags BPE by 0.165 BPB** (main-horse saw ~0.01 at similar FLOPs). The gap closes only slowly over training (0.19 → 0.17 → 0.165).

**23:20 ✅ eval sanity check.** Official pretrained `cartesia-ai/hnet_1stage_L` on our val: **0.7562 BPB**, L1/L0 0.205.
The paper reports 0.755 BPB and BPIC 4.8, so data, tokenization, and BPB accounting are correct, and the gap is in training dynamics.
Hypothesis: too few optimizer steps. main-horse's H-Nets used 22–50k-byte batches over 14–69k steps; ours had 2.3k steps.
→ testing batch 65k bytes (`S0-batch`).

**23:20–00:05 S0 batch test** (0.6B bytes; batch 8×8192 = 65k bytes, 9.2k steps; group `S0-batch`):
H-Net lr 2e-3 / 1e-3 → 1.3680 / 1.3738 (vs 1.3721 at 262k) · BPE lr 2e-3 → 1.2369 (vs 1.2076 at 262k).
BPE lr 4e-3 @262k → **1.1910** (BPE prefers higher LR).
❌ More optimizer steps don't close the gap (0.13–0.18 BPB). Keeping 262k-byte batches since they're faster.
The paper's own Fig. 3 also shows 1-stage H-Net behind BPE early in training (crossover after ~100B bytes),
so a BPE lead at 0.6–3B bytes may be genuine. main-horse's parity used training BPB on length-sorted FineWeb with his own FLOP accounting, so it isn't directly comparable.

**00:25 decision (user).** S0 result accepted as: *H-Net trails compute-matched BPE at this scale (~0.17 BPB @0.6B bytes), consistent with the paper's late crossover*.
Next, test the within-byte-level claims (DC > pool > byte-isotropic). GPU time goes to the latent ladder,
judged against our own H-Net baseline at the same budget and seed. S0 H-Net @0.6B: wd 0 → 1.3875 ❌ (keep wd 0.1).

**00:58 S0-short summary** (0.6B bytes, ~4.4e16 FLOPs, best LR each, val BPB):
| model | params | best LR | val BPB |
|---|---|---|---|
| BPE Transformer d512·T8 | 77M (51M emb/head) | 4e-3 (8e-3 → 1.567 unstable) | **1.1910** |
| byte Transformer d384·T5, SWA1024 | 9M | 4e-3 (2e-3 → 1.3563) | **1.3473** |
| H-Net 1-stage DC, N=6 | 33M | 2e-3 | 1.3721 |
| H-Net pool k=6 | 37M | 2e-3 | 1.3753 |
❌ At this budget the paper's ordering does **not** reproduce: the plain byte Transformer beats H-Net by 0.025, and DC beats pool by only 0.003 (noise TBD).
Likely cause: H-Net is undertrained at 0.6B bytes (its main net sees ~100M chunks). The 3B-byte runs will test this.

**01:13 noise.** H-Net @0.6B with seed 1 → 1.3744 (seed 0: 1.3721), so differences < ~0.005 BPB are noise (DC vs pool is a tie at 0.6B).

**01:28 ✅ S1a `s1a_tf_online`** (+ L_TF = L1(head(P(c≤j)), sg(c_{j+1})), online stop-grad targets, β=1): **1.3502 val BPB**,
−0.022 vs the H-Net baseline (≈8× noise). Side effect: chunks became word-aligned (`|Bromide| in| the |Allegheny| River|`)
and compression relaxed 1/6 → 1/5.1 (+6% FLOPs, too small to explain the gain alone).
Latent: val tf_l1 0.485 vs copy baseline 0.804, cos 0.56, target eff. rank 195/256 (no collapse).

**01:44 S1b `s1b_tf_ema`** (EMA target encoder, τ 0.996→1 cosine): 1.3612. ✅ beats baseline (−0.011), but ❌ loses to online targets (1.3502).
In a short run the encoder moves fast, so EMA targets go stale. → Ladder continues with **online** targets; EMA τ=0.99 retest queued.

**01:59 ✅ compression control** `s1c_hnet_N5_control` (plain H-Net, N=5): 1.3809, *worse* than N=6 (1.3721).
So S1a's gain is **not** from its lower compression; it comes from the latent objective.

**02:27 ❌ S2 `s2_tf_r_online`** (+ 2-step rollout L_R, β_r=1, gradients everywhere): 1.3707 ≈ baseline, worse than S1a.
TF cosine 0.56 → 0.36, word-aligned chunks lost, 2× step time.
Hypothesis: the rollout gradient reaches the encoder (V-JEPA 2-AC freezes it). Next: β_r=0.25 (`s2b`) and predictor-only rollout gradients (`s2c`).

**02:40 S3 control `s3c_zero_main`** (decoder gets zeros instead of main-net output): 1.4106.
Insight: at 0.6B bytes the main network adds only **0.038 BPB** to plain H-Net (1.3721), so it is barely used.
That explains H-Net losing to the byte Transformer here. With S1a's latent loss it adds 0.060 BPB (+60%).

**02:57 ✅ S1b' `s1b2_tf_ema099`** (EMA targets, τ=0.99 constant): **1.3470**, best so far (≈ online 1.3502 within noise). Eff. rank 208, cos 0.54.
⇒ The slow τ schedule (0.996→1) was the problem, not EMA itself. **Adopt EMA τ=0.99** (it matches the final V-JEPA design).

**03:26 ❌ S2b `s2b_tf_r025`** (rollout β_r=0.25): 1.3757. Even a small rollout weight erases the S1 gain (TF cos 0.49, chunks no longer word-aligned).
✅ `tests/test_rollout_grad.py`: rollout gradients with the mem-efficient SDPA kernel match the math kernel (identical loss, ≤4% rel. grad diff, bf16 noise), so this is not a bug.
Hypothesis: with a trainable context encoder, L_R pushes chunk reps toward being easy to roll out. That costs byte-level info. `s2c` (predictor-only rollout grads) tests it.

**03:56 ✅ S2c `s2c_tf_r_predonly`** (rollout gradients train only the predictor path: cached context + fed-back ĉ detached): **1.3524** (≈ S1a, BPB preserved).
Latent much better: TF cos **0.73** (S1a 0.56), tf_l1 0.451, rollout L1 0.52 (S2: 0.80). Word-aligned chunks back. Eff. rank 175.
⇒ Hypothesis confirmed: rollout must not train the context encoder (as in V-JEPA 2-AC, frozen encoder). **Adopted.**
Next: `s2d` = EMA 0.99 + L_TF + predictor-only L_R (full joint recipe), then `s3a` = same with the main net cut off from the AR loss.

**04:11 S1 β sweep** (online targets): β_tf=0.3 → **1.3425** BPB (best LM number so far) but TF cos 0.32. β=1 → 1.3502, cos 0.56.
⚠️ BPB vs latent-quality trade-off. The final latent recipe keeps β=1; β=0.3 is the best setting if H-Net is used only as an LM.
β_tf=3 → 1.3917 ❌ (worse than baseline). Optimal β for BPB is ≈0.3–1.

**04:55 ❌? S2d `s2d_ema_tf_r_predonly`** (EMA 0.99 + L_TF + predictor-only L_R): 1.3648, worse than its parts (EMA only 1.3470; online+predonly-rollout 1.3524).
TF cos 0.42, chunks not word-aligned. Observation: the good runs (S1a, S2c) developed word-aligned chunks and the weak ones didn't.
This may be a stochastic bifurcation. → seed-1 replicates of S2d and S2c before choosing the 3B recipe.

**05:05 chunk statistics** (`scripts/chunk_stats.py`; edge precision = share of boundaries at or right after whitespace, chance ≈0.3; space recall = share of spaces with a boundary at w/w+1):
baseline H-Net 0.21–0.33 / 0.20–0.29 (≤ chance, mid-word) · S1a 0.62/0.67 · **S2c 0.78/0.87** · S1b 0.55/0.59 · S1b' 0.36/0.38 · β0.3 0.14/0.15 · S2d 0.27/0.28.
⇒ The latent loss *can* induce word-like chunks (as in large pretrained H-Nets). Word alignment tracks latent cos, but **not** BPB.
The BPB gain from L_TF holds across all 6 latent runs (1.343–1.365 vs baseline 1.372–1.374).

**05:23 ❌ S3a `s3a_latent_main`** (main net trained only by latent losses; decoder reads its detached raw output): 1.4213, *worse* than the zero-main control (1.4106).
The latent predictor itself learns (TF cos 0.41, same as joint S2d), but the decoder can't use the raw truncated output z[:, :256].
Fix (design): the decoder should read the **prediction** ĉ_{j+1} = head(P(c≤j)), which estimates the encoder state summarising
exactly the bytes it must emit. Added `latent.decode_from_pred`. Queued `s3b` (latent-only + decode-from-pred) and `s3d` (joint + decode-from-pred).

**05:55 S2d seed 1** → 1.3676 (seed 0: 1.3648). EMA + predictor-only rollout is reproducibly worse (~1.366) than either part alone. Not a bifurcation.

**06:24 ⚠️ S2c seed 1** → 1.3776 (seed 0: 1.3524). Seed 1 did *not* find word-aligned chunks (edge prec 0.49 vs 0.78; cos 0.46 vs 0.73).
⇒ Rollout runs bifurcate. Seed means: S2c 1.365, S2d 1.366 (baseline 1.373). Rollout adds variance but no average BPB gain.
L_TF alone is robust: 1.3425–1.3502 over 4 settings (S1a, S1b', β0.3, + S1b 1.3612). Seed replicate of S1b' queued.

**06:51 ✅ S3b `s3b_latent_decode_pred`** (latent-only main net; decoder reads detached head(P(c≤j)) = predicted next-chunk latent): **1.4037**.
That beats zero-main (1.4106) and S3a (1.4213), so the latent predictor's output becomes usable by the byte decoder (+0.007 BPB of info).
Latent quality is up (TF cos 0.58 vs 0.41 for S3a/S2d), with word-aligned chunks (0.75/0.69).
**Cost of cutting the top half** at 0.6B ≈ 0.04 BPB vs the joint model (S2d 1.365). Joint main-net info = 0.046 BPB vs latent-only = 0.007.

**07:19 S3d `s3d_joint_decode_pred`** (joint, decoder reads head(P(c≤j))): 1.396, TF cos 0.68. Decomposition of the latent-style model's cost @0.6B:
joint, raw z (S2d) 1.365 → **+0.031** decoder reads predicted latent (S3d) 1.396 → **+0.008** predictor trained by latent losses only (S3b) 1.404 → zero-context control 1.411.
⇒ Most of the cost is the latent bottleneck. Removing AR gradients from the predictor costs only 0.008 BPB.

**08:28 S0 @3B** `s0_hnet1s_256-512_m4T9m4` (11.4k steps, ~2.2e17 FLOPs): **1.2750** val BPB (0.6B: 1.372).

**08:40 sample-efficiency estimate.** Fit H-Net baseline BPB(D) = E + A·D^−α through the cooled points (0.6B: 1.373 mean of 2 seeds; 3B: 1.275). α ≈ 0.10–0.14 for E ∈ [0.7, 0.9].
Data the baseline needs to match: S1a 1.350 → 0.86B (**1.4×**) · S1b' 1.347 → 0.90B (**1.5×**) · β0.3 1.343 → 0.96B (**1.6×**).
Overhead: +3–8% wall/FLOPs, so ≈1.4× compute efficiency (~70% of training time for baseline quality).
⚠️ Measured at 33M params / 0.6B bytes only. Likely shrinks with scale (small H-Nets underuse the main net). The f3 @3B run is the first test.

**08:50 GOAL (user): hillclimb the dynamic-chunking architecture until 10 AM PST (18:00 UTC) and show strong results.**
Protocol: 0.6B-byte runs (~15 min), one change at a time vs the current best; FLOP-matched architecture changes; final best confirmed at 3B against the H-Net 3B baseline (1.2750).
New knobs: `latent.target_norm=ln`, `latent.extra_horizons=[k..]` (+`beta_h`), `latent.head=mlp`.

**09:04 S0 @3B** `s0_bytetf_d384T5`: **1.2325** vs H-Net 1.2750. ❌ The byte Transformer's lead grows with data (0.025 @0.6B → 0.043 @3B).
At 0.6B, H-Net + latent loss (1.343–1.347) already ties the byte Transformer (1.347).
⇒ **Hillclimb target: dynamic chunking > plain byte Transformer.** Suspected weak spot: H-Net's byte-level enc/dec (Mamba, no MLP, d=256).
⚠️ d_model 320/384 for stage 0 is not usable: Mamba2 in_proj width must be a multiple of 8 for causal_conv1d with seq_idx (needs nheads%8==0, headdim fixed at 64). Replaced with an N=4 variant.

**09:22 ✅ S1b' seed 1** → 1.3529 (seed 0: 1.3470). Mean **1.350 ± 0.003** vs baseline mean 1.373. The latent-loss gain (−0.023) is robust across seeds.

**09:40 ✅✅ HC-A1 `hcA1_ema_b03`** (EMA 0.99 + β_tf 0.3): **1.3224**. New best: −0.051 vs H-Net baseline, −0.025 vs EMA β=1.
At equal FLOPs (4.55e16) it beats the byte Transformer (1.3473). Chunks not word-aligned (0.16). → `configs/hc_best.yaml` = this; round B builds on it.
**09:56 ❌ HC-A2 `hcA2_ema_lntgt`** (layer-norm targets, β 1): 1.3573 vs base 1.350 ± 0.003. Rejected.
**10:12 ❌ HC-A3 `hcA3_ema_h2`** (+ extra head predicting c̄_{j+2}, β_h 0.5, β 1): 1.3623 vs base 1.350. Rejected. Like A2 and β=3, more latent pressure hurts BPB.

**10:28 ✅✅✅ HC-A4 `hcA4_ema_mlphead`** (EMA 0.99, β 1, **MLP predictor head** D1→2D1→D0 GELU): **1.3029**. New best: −0.047 vs the linear head, −0.070 vs H-Net baseline.
FLOPs 4.62e16 (~equal to the byte Transformer's 4.55e16 → 1.3473). Interpretation: the MLP head absorbs the latent mapping, so the main net's output space is less constrained for the AR path.
❌ Round B launch failed: overrides with spaces get split by xargs. Relaunched on the new `hc_best` (MLP head) without spaces, together with the MLP seed replicate and MLP + β 0.3.
⚠️ Compute accounting: logged `train_flops` excludes the EMA target-encoder forward (~5.2 MFLOP/byte, +7%) and the predictor head (<1%).
True cost of hcA4 ≈ 4.9e16 vs the byte Transformer's 4.55e16 (+8%), which is small next to its 0.044-BPB margin.
Wall-clock is a different story: H-Net runs take ~15–16 min vs 7 min for the byte Transformer at equal FLOPs (Mamba/varlen kernels at lower MFU).

**10:45 ⚠️ HC-C1 `hcC1_best_seed1`** (EMA, β 0.3, linear, seed 1): 1.3486 vs seed 0's 1.3224. Mean **1.3355**. Latent runs have seed spread up to ~0.026, so every claimed win needs a replicate.
Seed means so far: baseline 1.373 · EMA β1 linear 1.350 · EMA β0.3 linear 1.336.

**11:01 ⚠️ HC-C3 `hcC3_mlp_seed1`** → 1.3408 (seed 0: 1.3029). MLP head mean **1.322** (best recipe on average; −0.051 vs baseline, −0.025 vs byte Transformer).
**Variance source found:** over 18 latent runs, corr(val L1/L0, BPB) = **−0.89**. The latent loss pushes the router past its 1/6 target (0.18–0.215), and seeds that overshoot more win.
Plain H-Net at N=5 does *not* benefit (1.381), so the gain needs latent loss + more chunks together. → Round D: MLP head + N=5 (2 seeds) to make that explicit and stable.
**11:17 ❌ HC-C4 `hcC4_mlp_b03`** (MLP head, β 0.3): 1.3553 vs MLP β 1 mean 1.322. With an MLP head, β 1 is better. Rejected.

**11:34 HC-D1 `hcD1_mlp_N5`** (MLP head + ratio target N=5): **1.2851**, but L1/L0 0.26, so FLOPs 5.25e16 (+15% vs byte Transformer, +24% vs H-Net). Not FLOP-fair yet.
→ FLOP-matched variant: main T9→T7 (`hcD3/4`, 2 seeds).

**11:50 HC-D2 `hcD2_mlp_N5_seed1`** → 1.3203 (seed 0: 1.2851) at equal compression (0.256 vs 0.260), so compression isn't the only variance source.
✅ **Paired-seed analysis:** seed 1 is worse than seed 0 for every latent recipe, from the first evals onward (no spikes). The **ranking is identical across seeds**:
| recipe | seed 0 | seed 1 | mean Δ vs H-Net |
|---|---|---|---|
| H-Net baseline | 1.3721 | 1.3744 | — |
| EMA lin β1 | 1.3470 | 1.3529 | −0.023 |
| EMA lin β0.3 | 1.3224 | 1.3486 | −0.038 |
| EMA **MLP** β1 | 1.3029 | 1.3408 | **−0.051** |
| EMA MLP β1 N=5 (+24% FLOPs) | 1.2851 | 1.3203 | −0.070 |
⇒ Every hillclimb step is confirmed paired by seed. The seed only changes the magnitude.

**12:06 ✅✅ HC-D3 `hcD3_mlp_N5_T7`** (MLP head, N=5, main T9→T7, FLOP-matched): **1.2733** @4.70e16 FLOPs, 26.5M params.
−0.099 vs H-Net baseline (paired seed 0), −0.074 vs byte Transformer (4.55e16). Beats the deeper N5-T9 (1.2851) with less compute.
Chunks ~3.6 bytes, space recall 0.87. Direction: more and smaller chunks + shallower main + latent loss → round E (N4·T5, N3·T4).
**12:23 ✅ HC-D4 `hcD4_mlp_N5_T7_seed1`** → 1.2970 (seed 0: 1.2733). Mean **1.285**. Δ vs H-Net (paired): −0.099 / −0.077 (mean −0.088). Vs byte Transformer: −0.062. Seed-1 ranking preserved.
**12:38 ❌ HC-E1 `hcE1_mlp_N4_T5`** (N=4, main T5): 1.3111 vs hcD3 1.2733 (paired seed 0). Too shallow a main net / too fine chunks. N=5·T7 stays best.
**12:55 HC-E2 `hcE2_mlp_N3_T4`** (N=3, main T4): 1.2764 vs hcD3 1.2733 (paired seed 0, tie within noise). See FLOPs below; N=5·T7 is kept as final (launch_final picks the lowest BPB).

**13:09 ✅ control `hcD5_plain_N5_T7`** (same final architecture, no latent loss): 1.3619. Architecture alone gives −0.010 vs H-Net (1.3721).
The latent loss on top gives −0.089 (→ 1.2733), so the gain comes from the latent objective, not the architecture.

**14:29 ✅✅✅ FINAL @3B `f3_D3_mlp_N5_T7_s0`** (EMA 0.99 + MLP head + N=5 + main T7): **1.1057 val BPB**.
H-Net 3B: 1.2750 (−0.169) · byte Transformer 3B: 1.2325 (−0.127). FLOPs 2.67e17 (+18% vs bytes, +28% vs H-Net; the router went to L1/L0 0.30 at 3B) +7% EMA.
Mid-training it already led by 0.10–0.15 at equal steps.
✅ **Causality verified** (`tests/test_causality2.py`): perturbing future bytes leaves the loss on earlier positions unchanged (0.83012 vs 0.83006). Residual logit wiggle (≤0.25) is bf16 packing noise, identical for the plain H-Net checkpoint.
Rough compute-efficiency (log-linear in compute along each baseline's 0.6B→3B curve): the byte Transformer would need ~5× and H-Net ~15× more compute to reach 1.106.
Queued: BPE @3B (parity check) and the no-latent control @3B.
**15:51 ✅ FINAL @3B seed 1** `f3_D3_mlp_N5_T7_s1`: **1.1081** (seed 0: 1.1057). Mean **1.107 ± 0.001**, far tighter than at 0.6B. Robust.

**16:34 BPE @3B** `s0_bpe_d512T8` (lr 4e-3): **1.0815**. Gap to BPE at 3B: plain H-Net 0.194 → final latent H-Net **0.025**.
⇒ The latent recipe closes **~87% of the H-Net→BPE gap** at this scale. Tokenizer-free BPE parity isn't reached yet, but it's close.

**17:45 ⚠️ control @3B `f3_plain_N5_T7_s0`** (final architecture N=5·T7, **no** latent loss): **1.1361**. This changes the 3B attribution:
| @3B (val BPB) | H-Net N6·T9 | H-Net N5·T7 (no latent) | + latent loss (final, 2 seeds) | byte Transformer | BPE |
|---|---|---|---|---|---|
| | 1.2750 | 1.1361 | **1.1069** (1.1057 / 1.1081) | 1.2325 | 1.0815 |
Decomposition of the −0.168 gain vs H-Net @3B: architecture/compression (N=5, shallower main) **−0.139**, latent loss **−0.029** (paired seed 0).
At 0.6B it was the reverse (arch −0.010, latent −0.089). So the latent loss mainly speeds early learning and still helps at 3B. The finer-chunk architecture pays off with more data.
Note: the plain N5·T7 router also drifts to L1/L0 ≈ 0.29 at 3B, so its FLOPs rise like the latent model's.

## 2026-11-10

**Ladder code (no GPU runs yet).** Everything in `NEXT_STEPS.md` is implemented: one config per rung in `configs/ladder/`,
gates and phase-winner selection in `scripts/ladder.py`, a multi-GPU queue with asynchronous gates (`scripts/run_jobs.py`),
a GPU preflight (`scripts/preflight.py`), data-parallel training (torchrun), early kill against a reference run, exact
per-batch FLOPs (now including the EMA target encoder, rollout passes and nested levels), exact k-step rollout, nested
levels with top-down, the surprise / identity / STE / REINFORCE routers, the eval suite and the latent chain-of-thought
evaluation. 34 CPU tests (`tests/cpu`, official H-Net modules with pure-PyTorch kernel stubs) cover equivalence with the
official forward, causality of every variant, gradient wiring, rollout exactness, the trainer (incl. 2-process DDP),
gates and job files. ⚠️ R0 re-runs the hcD3 recipe first to confirm the refactor reproduces 1.2733 / 1.2970.

**Ladder run on 2× H100 (Lambda), 21:42 UTC.**
- **Preflight caught three bugs:**
  - **Level-1 rollout under untied nesting.** It read the wrong qkv capture tag, so all of phases 4–5 would have crashed.
  - **Preflight's encoder-gradient check ran at init.** There the zero-initialised residual projection blocks the gradient, so R4/R5 failed falsely and R6's detach check passed trivially.
  - **A CPU test deleted the live preflight stubs.**

  All three are fixed (e7eedde), and every config passes preflight on GPU.
- **Speed.** 0.24 s/step, 9.9 min for a 0.6B run: ~1.7× the A100.
- **22:12 ❌ gate R0 (hardware, not code).** The new code gave 1.2872 / 1.3148 vs 1.2733 / 1.2970 (Δ +0.016 > 0.01). The configs are identical and the curves match at step 0; the router settled at L1/L0 ≈ 0.25 instead of ≈ 0.27.

  The old code (48cd658) re-run on the same H100s gives **1.3131 / 1.3245** with L1/L0 0.25 / 0.23. So the A100 reference can't be reproduced on H100 by either code, and on the same hardware the new code is not worse (−0.018).

  The cause is H100 numerics in the router: fewer chunks, so less main-net compute. Every ladder comparison is H100 vs H100 and paired by seed, so the ladder resumes at R1 with L-R0-s* as the reference (`logs/jobs/phase1_resume.txt`). It adds a same-seed replicate L-R0rep-s0 to measure run-to-run noise against the 0.01 gate tolerances.
- **Latent loss vs none at 0.6B (seed 0).** R0 1.2872 vs the old no-latent control 1.3619 (old code, A100): −0.075, vs −0.089 before. Needs an H100 no-latent run to be paired.
- **22:42 side track: `side-f3-plain-3p2B-s0` stopped at 6000/12207** (`runs/<name>/KILL`). Val was flat at 1.4897 → 1.4888 between steps 2000 and 4000 while its twin `side-f3-plain-s1` went 1.4721 → 1.3035. Train BPB jumped at ~2500 and the router stuck at L1/L0 ≈ 0.19. It reached 1.4397 at 6000. Rerun queued as `side-f3-plain-3p2B-s0-r2` after the ladder.
- **22:58 ❌→↻ R1-s1 false early kill.** It was killed at step 1000 (44%): 1.6149 vs R0-s1 1.5507 (+0.064 > 0.03). But R1-s0 was *better* than R0-s0 at that step (1.575 vs 1.5909), and R0's own seeds swap order between step 1000 (1.5909 / 1.5507) and the end (1.2872 / 1.3148). Mid-training val BPB on H100 swings 0.05–0.07, so the kill was noise.

  Fixes:
  - Default `kill_after` is now 0.8 (`KILL_AFTER` in train.py), so only the late eval can kill.
  - R1-s1 is rerun (`logs/jobs/r1s1_rerun.txt`). Try 1's summary was set aside as `summary.killed_try1.json`, so gate R1 waits for the rerun instead of failing.
- **23:20 run-to-run noise measured → gates widened (user decision).** `L-R0rep-s0`, an identical rerun of R0-s0, ended at **1.2733** vs 1.2872. It had been 1.5282 vs 1.5909 at step 1000 and 1.3203 vs 1.3352 at step 2000.
  - The val eval is deterministic, so this is training-trajectory noise: GPU non-determinism amplified by the router and latent-loss dynamics.
  - So the R0 gate failure (+0.016) was within noise, and the 0.01 / 0.0 neutral-rung tolerances were coin flips.
  - Changes:
    - Neutral rungs now pass at Δ ≤ +0.03 (`TOL`) and tf/copy ≤ 1.10× (`RATIO_TOL`). R4 (a costly change) passes at Δ ≤ +0.05.
    - Early kills need a gap of ≥ 0.05 (`KILL_MARGIN_MIN`) after 80% of training.
    - Signal checks are unchanged: trunk info > 0.01 vs the floor, and beats copy.
  - R1-s0 try 1 had been killed at step 2000 (+0.036 vs R0-s0). It is being rerun to completion with no early kill (`logs/jobs/r1s0_rerun.txt`), so gate R1 decides on final values.
- **23:22 the 10B side pair OOMed at launch** (`side-f10-*`: 4 jobs on a GPU). They are requeued with the new guard (`RUN_JOBS_MAX_PER_GPU=1 RUN_JOBS_MIN_FREE_GB=30`).
- **23:29 R1-s1 rerun → 1.3065, vs R0-s1 1.3148 (−0.008).** The first try was early-killed at step 1000 as +0.064, which confirms that mid-training gaps are noise. R1-s0 is being rerun; gate R1 waits for it.
- **23:33 `side-f3-plain-s1`: 3B, no latent loss, H100 → 1.1169.**
  - The old no-latent control (seed 0, A100) was 1.1361. The joint-latent 3B runs (A100) are 1.1057 / 1.1081.
  - Paired by seed 1, the latent gain is only −0.009, vs −0.030 on seed 0. The no-latent seeds differ by 0.019, unlike the latent seeds (0.0024), but hardware and code also differ.
  - A clean H100 pair is queued: `side-f3-latent-s1` (`logs/jobs/side_f3_latent.txt`).
- **00:05 R2-s0 (the decoder reads the predicted latent) → 1.3509.** FLOOR-s0 (decoder alone) is 1.5100, so **trunk info = 0.159 BPB**; the gate needs > 0.01. tf/copy 0.59. This costs +0.038 vs R1-s0, as expected for a costly rung. The first latent-only signal is strong.
- **00:04 normal H-Net, 0.6B, seed 0, H100 → 1.4007** (A100: 1.3721). Plain N5·T7 reproduced exactly at 1.3619. The latent R0 (1.287 / 1.273) beats the H100 H-Net by ~0.12.
- **00:07 scheduling.** Two guarded queues saw the same free GPU at once: a new job is invisible to nvidia-smi for 10–30 s. That started both 10B runs and the 3B latent run together on one GPU.
  - The guard now takes a lock and leaves 2-minute launch reservations.
  - The 10B pair (~7.5 h while sharing) is paused until the phase-3 hold, when the GPUs would otherwise wait for the user's review. `side-f3-latent-s1` keeps running.
- **00:58 H100 0.6B baselines complete** (paired by seed, same hardware and code):

  | 0.6B, val BPB | s0 | s1 | mean |
  |---|---|---|---|
  | normal H-Net (N6·T9) | 1.4007 | 1.3729 | 1.3868 |
  | same arch, no latent (N5·T7) | 1.3619 | 1.3786 | 1.3702 |
  | latent R0 (N5·T7) | 1.2872 (rerun 1.2733) | 1.3148 | 1.301 |

  The latent loss is worth **−0.069** against the same architecture and **−0.086** against H-Net (both seeds agree), about 4× the run-to-run noise. Same as the A100 result.
- **00:52 gate R3 passed:** trunk info 0.127, tf/copy 0.61. The main net is trained only by latent prediction.
- **00:31 L-FLOOR3B-s0 → 1.4092.**
- **01:41 clean 3B pair (H100, seed 1, same code): latent `side-f3-latent-s1` 1.1088 vs no-latent `side-f3-plain-s1` 1.1169 → −0.008 BPB, at +10.6% FLOPs.**
  - The latent run reproduces the A100 seed-1 result (1.1081), so the old no-latent seed-0 control (1.1361) was probably a bad run, and "−0.030 at 3B" overstated the gain.
  - Compute-adjusted, the no-latent curve gains ~0.097 BPB per doubling, so +10.6% FLOPs is worth ~0.014. On seed 1 that is more than the latent gain.
  - **At 3B the latent loss gives no clear compute-adjusted gain yet.** At 0.6B it was −0.069 (≈4× noise).
  - To settle it: an H100 no-latent seed 0, and the compute-matched control (`side-f3-plain-3p2B-s0-r2`).
- **01:31 gate R6 passed (go/no-go):** the fully latent-only trunk (no byte gradient into encoder, router or main net) keeps trunk info 0.057 (R5: ~0.12), tf/copy 0.59, and no collapse (within 0.994, eff. rank 227). M1 (R6 @3B) is running.
- **02:57 ✅ Phase 1 complete: gate M1 passed.** R6 at 3B (fully latent-only trunk: no byte gradient into encoder, router or main net). Readout 1.2989 / 1.2900 vs FLOOR3B 1.4092 → **trunk info 0.115**; it was 0.057 at 0.6B, so it grows with data. tf/copy 0.547. Phase 2 (routers R7–R9) starts. The queue's `L-R1-s*:3` entries are the first-try kills; the reruns passed.
- **03:18 3B latent vs no-latent, corrected with two more controls (H100, same code):**

  | 3B, val BPB | run(s) | mean |
  |---|---|---|
  | latent | 1.1057 (s0, A100) · 1.1081 (s1, A100) · 1.1088 (s1, H100) | 1.1075 |
  | no-latent | 1.1341 (s0, H100; A100 was 1.1361) · 1.1169 (s1) · 1.1246 (s1, compute-matched: 3.32B bytes, 2.67e17 FLOPs) | 1.1252 |

  - **The latent gain at 3B is ≈ −0.017 (~1.5%).** Compute-matched on seed 1 it is −0.016.
  - The 01:41 note ("no clear compute-adjusted gain; 1.1361 probably a bad run") was wrong. The H100 rerun reproduces 1.1361, and no-latent 3B runs simply spread ~0.009 sd.
  - **The latent runs spread only ~0.002:** the latent loss also makes training much more reproducible.
  - The gain shrinks with data (−0.069 at 0.6B) but is not gone. The 10B pair will tell whether it keeps shrinking.
- **03:13–03:17 R7 (identity router) killed on both seeds at step 2000:** +0.10 vs R6 (~5× noise). Boundaries were on target (0.25) and word-like (edge prec 0.55–0.70), yet worse than fixed chunks under latent-only training. Phase 2 continues with R8 and R9.
- **03:39 phase 2 select → R9 (prediction-error boundaries).** R7 (identity router) and R8 (STE router) were killed on both seeds at 87%: +0.10 to +0.17 vs R6. R8 also collapsed: target eff. rank ~40, tf/copy > 1 by mid-training. R9 passed at +0.023.
  - Fix before select (e7e0ecd): R8/R9 were judged against R7, which was killed, so they failed automatically. All phase-2 rungs replace R6's fixed chunks, so the reference now falls back to R6.
  - **3rd seed (R9 passed within noise of its threshold):** R9 = 1.4664 / 1.4795 / 1.4592 (mean 1.4684) vs R6 1.4508 / 1.4482 → **+0.019**, confirmed.
  - So under latent-only training, boundaries set by latent prediction error cost ~0.02 vs fixed 4-byte chunks. Gradient-trained routers fail.
  - M2 (R9 @3B) is running.
- **04:40 ✅ Phase 2 complete: gate M2 passed** (R9 @3B: readout 1.3459 / 1.3505, trunk info 0.061, tf/copy 0.51). The ladder is **on hold before phase 3** (`logs/hold_P3`) until the user reviews the benchmarks. The 10B latent / no-latent pair restarted at 04:40.
- **Phase-2 benchmarks** (`scripts/eval_suite.py`, standard H-Net/ATDC/DeepMem zero-shot set; report: https://claude.ai/artifact/P1egydDegMKeRhXQ76hpc8). At 3B:
  - **7-task average (± ~0.5):**

    | model | 7-task avg |
    |---|---|
    | BPE | 35.6 |
    | joint latent | 33.1–33.6 |
    | no latent | 33.0–33.2 |
    | byte Transformer | 32.1 |
    | latent-only M1 | 31.1–31.8 |
    | H-Net N6·T9 | 31.6 |
    | latent-only M2 | 31.1 |
    | floor | 30.7 |

    The latent loss does not move the average. It helps on LAMBADA (15.7–16.4 vs 14.4–14.9), SciQ (+1) and WikiText BPB (1.32–1.33 vs 1.36–1.39).
  - **BPE:** best on ARC-e (43.4) and the average, but worst on SciQ (69.1) and much less robust to character noise: +2.25 BPB under the BLT noises vs ~1.6 for byte models.
  - **Latent-only trunks:** reach the H-Net N6·T9 baseline's average, though not the joint models' BPB. Their predicted latents are as good a next-word probe as the joint model's (M2 42.5–45.0 vs 43.8–45.6; floor 13.2). The prediction-error boundaries (M2) give the best morpheme F1 of any model: 25.1–25.4 vs 13–17 for the joint and fixed-chunk models.
- **07:42 ❌ 10B pair: the joint latent loss hurts with more data.** `side-f10-latent-s0` reached **1.0458** vs `side-f10-plain-s0` **1.0318** (+0.014 worse) with 1.17× the FLOPs. The gap by step went −0.028 (4k), +0.001 (8k), +0.004 (12k), +0.008 (20k), +0.010 (24k), +0.013 (32k), +0.015 (end). One seed, but monotonic.
  - Across scales: −0.069 at 0.6B, −0.017 at 3B, **+0.014 at 10B**. The latent auxiliary loss accelerates early training and then costs quality.
  - The no-latent H-Net at 10B (1.0318) beats the 3B BPE baseline (1.0815).
- **07:45 CoT study, phase 1 of the latent-reasoning plan.** Synthetic arithmetic (2–6 steps in training, 2–8 at test, answers mod 100, chance ≈1%). All runs are fine-tuned from H100 3B seed-1 checkpoints. Scripts: `latents/think.py`, `scripts/think_train.py`, `scripts/latent_cot.py`.

  | run | method | acc | 2–6 steps | 7 | 8 | seq. steps | MFLOP |
  |---|---|---|---|---|---|---|---|
  | B1 normal H-Net, written CoT | byte CoT | **94.3%** | 98–100 | 91 | 74 | 70 bytes + 18 main | 1644 |
  | B2 normal H-Net, direct | direct | 3.2% | 1–5 | 3 | 1 | 0 | 0 |
  | L1 latent H-Net, written CoT | byte CoT | **77.0%** | 95–100 | 42 | 2 | 68 + 11 | 1297 |
  | L1 | zero-shot thoughts (2–16, oracle 10.8) | 0–0.8% | | | | 2–16 main | 97–778 |
  | L2 thoughts from base (128k problems, 67 min) | thoughts (0–37) | 2.6–3.3% | | | | ≤37 main | ≤1788 |
  | L2b thoughts from L1 (64k problems) | thoughts | 1.5–2.8% | | | | | |

  - **Reasoning in latents failed:** trained thoughts sit at the no-reasoning level. L2 and L2b also lost written CoT (0%).
  - **Teacher-thoughts diagnostic** (true reasoning-chunk states fed as thoughts): L1 0.2%, L2 3.5% (= its no-thought 3.5%), L2b 3.5%. The answer path learned to *ignore* thoughts, so prediction error is not the whole story. Likely cause: byte CoT works because the byte decoder copies the last computed value from its own byte stream ("…=86." → "#### 86"), so the main net never had to carry the answer. Latent reasoning needs a latent → answer pathway that nothing ever trained.
  - **Side finding:** the latent-loss model's written CoT generalizes worse to longer chains: 42% vs 91% at 7 steps, 2% vs 74% at 8. Both are near-perfect within the training range.
  - **Next, simplest first:**
    1. Train answering from *true* reasoning latents with the CoT bytes removed (validates the read path).
    2. Then scheduled sampling from true to predicted thoughts.
    3. And/or make each thought decodable to its chunk's bytes, so it must carry the value.
    4. Fewer, more abstract thoughts (one per reasoning step).
- **07:55 Why L1 fails long chains: it stops writing early, it does not compute wrong.**
  - On 40 eight-step problems, B1 is right 30/40 and writes 16.5 of the 17 '=' signs; L1 is right 1/40 and writes 14.5.
  - Example: L1 writes 7 steps, then `#### <7th value>`. Within training lengths (2–6) both are 95–100%.
  - Hypothesis (one fine-tune per model): the latent loss sharpens prediction of the chunk where `####` begins, so a training-length prior on chain length takes over.
  - **Why the joint model is the wrong base for thoughts:** its decoder reads the main-net output, not the predicted latent. The latent-only models (R2+, M1/M2) decode *from* predicted latents, so the thought → text path is pretrained. Use them for the next attempt.
- **08:57 Latent thoughts on the latent-only model M1, pure loss** (`loss_pure`). The decoder is trained on stop-gradient latents and never shapes the thoughts; thoughts learn only from latent targets; every step also trains on the full written examples.

  | run | eval | acc | by steps 2–8 |
  |---|---|---|---|
  | T-base (M1 + written reasoning) | byte CoT | 74.3% | 100 100 100 100 98 19 2 (stops early) |
  | T-base | true latents | 0.5% | |
  | **T-read** (thought inputs always the true latents) | **true latents** | **34.0%** | 42 29 33 34 37 28 35 |
  | T-read | own thoughts (4 / 16 / oracle 15.6) | 1.5 / 3.0 / 4.3% | |
  | T-full (true → own over 60%) | own thoughts, oracle | 4.3% | |
  | T-full | true latents | 5.2% | |
  | any | no thoughts | ~3% | |

  - **Reading works:** with true reasoning latents and no reasoning bytes, 34% (vs 3%), flat across chain lengths, from a decoder that never shapes the latents. This is the first positive result on latent reasoning.
  - **Producing fails:** own thoughts ≈ chance. The thought L1 rises from 0.78 (teacher-forced) to 1.05 (own), so errors compound over ~16 thoughts of 4 bytes each. T-full also lost most of its reading (5%), because training late on uninformative own thoughts teaches the decoder to ignore them again.
  - **Next step:** first measure where own-thought rollouts break (error by thought index). Then try one thought per reasoning *step* ("b=42"): 2–8 thoughts instead of ~16, which is the temporal-abstraction idea. Keep some true-latent inputs (p_true > 0) to retain reading.
- **12:15 (2026-11-11) Why teacher-forced latent thoughts were stuck at 34%, and what fixed it (batch 1).**
  - **Setup.** M1 (latent-only), thoughts = the TRUE reasoning-chunk latents (teacher mode), answer decoded from latents only. Evaluated on all 2000 test problems.
  - **Training rules (user):**
    - non-contrastive objectives only
    - nothing hard-coded
    - always end-to-end, nothing frozen
    - the decoder gets its own loss, but its gradient never reaches the latents, encoder or trunk
    - teacher-forced on real encodings only
  - **New names:** hnet-cot = C-B1, latent-cot = C-L1, m1-cot = T-base, think-96k = T-read, think-1.6M = T-read2.
  - **Diagnostics added:**
    - `scripts/think_diag.py`: the decoder reads predicted / true / zero latents, with the answer fed in byte by byte.
    - `scripts/think_probe.py`: an MLP probe for the next answer byte from the true latent, the main-net output `z`, or the predicted latent.
    - `scripts/think_probe_value.py` and `scripts/think_geometry.py`.

  | run (all from think-96k unless noted; 1500 steps × 256) | acc | by steps 2–8 | decoder reads pred / true | probe true / z / pred |
  |---|---|---|---|---|
  | hnet-cot, written CoT (reference) | **94.6** | 100 100 100 100 100 93 71 | | |
  | latent-cot, written CoT | 75.9 | 100 98 96 95 97 44 1 | | |
  | m1-cot, written CoT | 72.8 | 100 100 100 100 99 9 1 | | |
  | think-96k (T-read) | 33.8 | 38 30 36 35 32 34 32 | 33 / 46 (n=200) | 99 / 35 / 31 |
  | think-1.6M (T-read, 1.6M problems total) | 50.5 | 63 54 52 48 53 44 39 | | |
  | think-dectrue (+ decoder also on true latents) | 51.8 | 54 58 53 52 48 47 50 | 52 / 81 | 86 / 41 / 38 |
  | think-dectrue-long (+3000 steps) | 41.2 | 48 42 43 42 45 36 33 | 41 / 80 | 19 / 17 / 5 |
  | think-l2 (L2 latent loss) | 38.4 | 38 37 41 37 37 40 38 | 38 / 68 | 78 / 31 / 25 |
  | **think-nopad** (padding fix + dec_true) | **70.2** | 71 76 77 65 66 68 67 | 70 / **99** | 98 / 60 / 53 |
  | think-ema999 (+ EMA tau 0.999) | 70.5 | 69 77 77 68 70 69 65 | 71 / 99 | 98 / 58 / 52 |
  | think-gamma5 (+ answer-latent loss ×5) | 66.3 | 70 73 73 62 62 65 60 | 67 / 97 | 94 / 53 / 46 |
  | **think-tgrad** (+ thought loss trains the encoder through the thought inputs) | **76.6** | 82 81 79 72 78 70 75 | 77 / 99 | 99 / 67 / 63 |

  - **Issue 1: the decoder never trained on real encodings.**
    - It only ever read the trunk's predictions, so it decoded true latents at 45% while a probe reads 98.7% from them.
    - Fix: `--dec_true` adds a decoder-only CE that reads the true next-chunk latents (target encoder, detached). 34% → 52%.
  - **Issue 2, the big one: padding counted in the latent losses.**
    - `think.py` pads each problem's row to the batch maximum. The byte losses masked that padding; the latent losses did not.
    - Real chunks were only 13% of the answer-latent loss terms and 54% of the full-document latent loss.
    - Fix: a `chunk_valid` mask (commit 6923bbd). A test shows every loss is identical with 0 or 128 extra padding bytes.
    - Results: decoder on true latents 81% → **99%**, on predicted latents 52% → 70%; accuracy 52% → **70%**. The encoder no longer drops digits (true-latent probe 86% → 98%).
    - With the bug, more training *hurt*: think-dectrue-long's true-latent probe fell to 19% and its accuracy to 41%. The encoder drifted to drop the hard-to-predict digits, and padding diluted the answer signal about 7×. That explains the think-1.6M plateau at about 47–50%.
  - **Issue 3: the predicted latent didn't carry the answer.**
    - Fix: `--thought_grad`. The teacher thought inputs keep their gradient, so the thought and answer latent losses also train the encoder through them. This is an end-to-end, missing-gradient fix.
    - Probes: `z` 60 → 67%, predicted latent 53 → 63%. Accuracy **76.6%** (82% at 2 steps), against 94.6% for written CoT.
    - The remaining errors are almost all in the units digit: the tens digit is 99.7%, the units digit 79%.
  - **No help:**
    - L2 instead of L1 (38%): the encoder drops more digit content (true-latent probe 78% vs 86%). Gradients were checked end to end: L2 reaches embeddings, encoder, main net and head at about 2.3× L1's norm, and the decoder's gradients are identical under both. An L2-from-scratch comparison is running.
    - Answer loss ×5 (66%).
    - Slower EMA target (70.5%, same as the baseline).
    - Contrastive InfoNCE: dropped by rule. Batch InfoNCE was also trivially solved at 99% by the context fingerprint.
    - Freezing the encoder: dropped by rule (end-to-end).
    - Distribution regularizers: dropped. It isn't collapse: RankMe 227 on FineWeb, and the task's low rank (about 26) just reflects the narrow data.
  - **Next:**
    - Combine thought_grad with the padding fix and train longer.
    - The units digit: the main net's `z` still holds only 67%.
    - Then own-thought rollouts at inference. Hierarchy comes later, via nesting.
