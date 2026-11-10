# Next steps

Updated Nov 10, 2026.

## Goal

The trunk (encoder, router, and a predictor at every level) learns only from latent prediction: each level predicts the latent of its next chunk. Bytes train only a readout decoder, which can't send gradient back into the trunk. Levels nest as deep as the data supports, and whether to go up another level is itself a routing decision.

Two losses stay because they constrain the model rather than carry information about the text: the ratio loss (how many chunks, i.e. compute) and an anti-collapse regularizer.

## How we climb

- **One change per rung.** Each rung is compared to the rung right before it.
- **0.6B bytes, 2 seeds, compared paired by seed.** Runs take ~15 min. Latent runs at 0.6B differ by up to ~0.025 BPB between seeds, but recipe rankings have held across seeds.
- **A neutral change** (expected to cost nothing) passes if the paired Δ is within ±0.01 BPB and the latent metrics don't get worse.
- **A costly change** (one that removes a signal) passes if the trunk still carries information (trunk info below stays clearly above 0) and the latent metrics hold. Log the cost.
- **If a rung fails,** try the alternative listed for it before moving up. Never stack a second change on a failing rung.
- **Milestones** get confirmed at 3B bytes.
- **Log every rung** in `logistics.md`.

## Metrics on every rung

- **readout BPB:** val BPB of the byte decoder.
- **floor BPB:** the same decoder with every trunk input zeroed, i.e. a small local byte model. Run once per decoder design.
- **trunk info = floor − readout.** This is the main number: how much the latent trunk actually tells the decoder.
- **latent quality:** L1 vs the copy baseline (`lat/copy_l1`), cosine, effective rank.
- **within-sequence variance of the chunk latents (new):** catches the document-ID shortcut (see design notes).
- **chunking:** compression L1/L0 and edge precision (`scripts/chunk_stats.py`).

## Phase 1: pure latent trunk, fixed boundaries

Start from the current best joint recipe and remove the byte signal one part of the stack at a time.

| Rung | One change | Pass if | Code |
|---|---|---|---|
| R0 | Reference: `hcD3` recipe (EMA 0.99, MLP head, N=5, main T7, joint). Add within-sequence variance logging. | logging only; reuse hcD3 / hcD4 (1.2733 / 1.2970) | small |
| R1 | Decoder gets its own byte embedding, added to its input. Also run the floor control. | neutral | small |
| R2 | Decoder reads the predicted latent $\hat c_{j+1}$ instead of the raw main-net output | cost logged (was +0.031 on the old recipe) | exists: `latent.decode_from_pred=true` |
| R3 | No byte gradient into the main net | cost logged (was +0.008 on the old recipe) | exists: `latent.detach_main=true` |
| R4 | Fixed boundaries, one every 4 bytes, instead of the router | close to R3 (fixed chunks tied learned ones at 0.6B in S0) | exists: `model.pool_k=4` |
| R5 | Add a within-sequence anti-collapse regularizer; byte gradient still reaches the encoder | neutral | new |
| R6 | No byte gradient into the encoder (detach the residual). The decoder is now a pure readout. | trunk info > 0, no shortcut | new |

**Milestone 1:** R6 at 3B, 2 seeds. The whole trunk now learns only from the latent loss, with fixed boundaries.

Notes:
- **R1** sets up R6. Once the encoder stops being trained for bytes, the decoder needs its own view of the current bytes.
- **R4** uses fixed chunks so the router problem (design notes) can wait until phase 2. With no router, the residual is the only byte path left into the encoder, so R6 is a single detach.
- **R6 is the go/no-go rung.** The expected failure is the document-ID shortcut. If it shows up, replace R5's regularizer with CPC-style negatives and rerun (R6b).

## Phase 2: learn the boundaries from the latent signal only

Each rung replaces R4's fixed chunks and is compared to R6. Keep the best.

From here on, the decoder path must also detach the boundary probability $p$. The dechunk smoothing and the STE both use $p$, so without the detach the byte loss would leak back into the encoder through the router. That keeps R6's guarantee intact; it isn't a separate change.

| Rung | One change | Pass if | Code |
|---|---|---|---|
| R7 | Option D: router with $q, k$ frozen at identity, so cuts fall where the latent-trained encoder changes direction; the ratio loss sets the count | compression stays near target; readout within R6 + 0.01 | small |
| R8 | Option A: trainable router, with gradient from an STE on the predictor input | better than R7 on readout or edge precision | small |
| R9 | Option C: cut where a small latent predictor is surprised | better than R7 | medium |

Option E (REINFORCE or Gumbel-sigmoid) only if all three fail.

**Milestone 2:** best of R7–R9 at 3B.

## Phase 3: longer horizons, one level

| Rung | One change | Pass if | Code |
|---|---|---|---|
| R10 | 2-step rollout loss with predictor-only gradients (the S2c recipe) | rollout error down, readout not worse | exists: `latent.beta_r`, `latent.rollout_predictor_only=true` |
| R11 | k-step rollout, k = 4, then 8 | error at step k stays below the copy baseline | extend `latents/rollout.py` |

Watch the seeds here: S2c split into a good and a bad outcome across seeds before.

## Phase 4: a second level

| Rung | One change | Pass if | Code |
|---|---|---|---|
| R12 | Add level 2 on top: encoder + predictor over level-1 chunks, fixed chunks of 3, its own latent loss. Nothing flows back down. | level-2 latent beats its copy baseline; level 1 unchanged | large: 2-level `forward_lm` |
| R13 | Top-down: level-2 predictions condition the level-1 predictor | level-1 latent error down, readout down | medium |
| R14 | Learned level-2 boundaries (best method from phase 2) | at least as good as R13 | small |
| R15 | Rollout at level 2 | as in R10–R11 | small |

Notes:
- **R12** is a passenger check: level 2 learns, but can't help or hurt level 1 yet.
- **R13** tests whether level 2 helps. The top-down path is trained only by level 1's latent loss, so this is still latent signals only.

**Milestone 3:** R15 at 3B.

## Phase 5: nest indefinitely

| Rung | One change | Pass if | Code |
|---|---|---|---|
| R16 | Tie weights across levels: one block reused at every level, plus a level embedding | within R15 + 0.01 | medium |
| R17 | Add level 3 using the shared block | level 3 helps level 2, as in R13 | small once R16 works |
| R18 | Choose depth by a fixed rule: keep going up while the top sequence is longer than M chunks | matches R17 at the same depth | small |
| R19 | Learned depth: above level 2, replace the fixed ratio target with a compute penalty, so a level can send nothing up where going up doesn't help | at equal compute, at least as good as R18; depth varies by content (code vs prose vs repetitive text) | medium |

How R19 turns "go up?" into a routing decision:
- The next level's router already picks which chunks go up.
- With a compute penalty instead of a fixed target, it can pick none in a region, so depth varies by region.
- The payoff for going up is still a latent signal: the drop in the latent loss of the level below.

Practical limit: 8KB windows. At 3–5× compression per level, 3–4 levels leave about 20 top-level chunks, so going deeper needs longer windows.

## Phase 6: use it

- **Latent chain of thought.** Roll the predictor forward k steps and decode only at the end.
  - On the 1-level model (~3.45 bytes per chunk), a latent step costs ~62 MFLOP vs ~101 MFLOP to spell the chunk out in bytes: 1.6× fewer FLOPs and ~3.45× fewer sequential steps. The step saving grows with each level.
  - Test on a synthetic multi-step task, e.g. arithmetic or graph reachability like Coconut's ProsQA.
  - Compare no chain of thought, byte-level chain of thought, and latent chain of thought, on accuracy vs FLOPs and vs sequential steps.
- **Benchmarks** at each milestone (list in the side track).

## Design notes

### Collapse vs shortcut

Stop-grad plus EMA already prevents collapse in our runs (target effective rank 175–208 of 256). SIGReg (LeJEPA) or VICReg would make that sturdier.

The risk without the byte loss is a **shortcut** instead:
- **What it looks like.** The encoder learns a code that is nearly constant within a document but differs across documents (a topic or document ID).
- **Why the regularizers miss it.** SIGReg and VICReg only constrain the spread *across the batch*, so this code passes them. Yet within a document it's trivially predictable: just copy the last chunk.
- **Detect it with** the within-sequence variance, and `lat/copy_l1` relative to the prediction error.
- **Fix 1 (R5):** apply the regularizer *within* sequences, e.g. on per-sequence-centered latents or on the differences $c_{j+1} - c_j$.
- **Fix 2 (R6b):** CPC-style InfoNCE with negatives drawn from the same window. The model then has to pick *which* chunk comes next, not just which document it's in.

### Where the router gets its gradient (checked in goombalab/hnet @ 3673fe1)

```
bytes -> encoder h_t -> router p_t = (1 - cos(q h_{t-1}, k h_t)) / 2 -> mask = p > 0.5
      -> chunks = h[mask]               (hard indexing: no gradient to p)
      -> main net
      -> dechunk smoothing uses p       (gradient to p, decoder path)
      -> x STE(p) + residual            (gradient to p, decoder path)
      -> decoder -> bytes
```

- **Every gradient to $p$ comes through the decoder,** plus the ratio loss, which only sets how *many* boundaries there are.
- **The latent loss reaches the encoder but not the router:** it trains the encoder's values through the chunk inputs, but never the boundary decision.
- **$q$ and $k$ start as the identity,** so a boundary means "the encoder state just changed direction." Reshaping $h$ therefore moves boundaries even with zero router gradient. That's probably how the word-aligned chunks appeared.

Options:
- **A. STE on the predictor input.** Feed $c_j \cdot \mathrm{STE}(p_{b_j})$ to the main net. The forward pass is unchanged, and the backward pass gives $\partial \mathcal{L}_{\text{lat}} / \partial p_{b_j} = \langle \partial \mathcal{L}_{\text{lat}} / \partial c'_j,\, c_j \rangle$, i.e. "does this chunk help prediction?" Limit: only existing boundaries get a signal.
- **B. Soft chunk inputs.** Build $c_j$ as a $p$-weighted average of the byte states in the chunk, so every byte's $p$ gets a gradient. Context side only: if the targets are soft too, the router can pick easy targets.
- **C. Prediction-error boundaries.** Cut where a small byte-level latent predictor is surprised. This is BLT's entropy patching moved into latent space, and it matches event-segmentation theory from cognitive science. No gradient through discrete choices at all. Warm up with fixed chunks.
- **D. Identity router.** Freeze $q$ and $k$ at identity and let the latent-trained encoder decide. The ratio loss sends gradient into $h$ through $p$.
- **E. REINFORCE or Gumbel-sigmoid** on the boundaries. General but noisy. Last resort.

Shared danger: the router can game the latent loss by cutting where prediction is easy, e.g. right after very predictable spans. Readout BPB, edge precision and the copy baseline keep it honest.

### The decoder as a readout

With all of its inputs detached, the decoder is a probe. Readout BPB then measures how much of the text the latent trunk keeps, the analogue of linear-probe accuracy in vision JEPA.

It needs its own byte embedding (R1). Under latent-only training, only the encoder states at boundary bytes ever get a gradient, so the per-byte residual stops being a reliable view of the current bytes.

## Side track: is the current joint result real?

Run these whenever the GPU is free; they don't block the ladder.

| @3B bytes | val BPB |
|---|---|
| H-Net N6·T9 | 1.275 |
| H-Net N5·T7, no latent loss (1 seed) | 1.136 |
| + latent loss (2 seeds) | 1.107 |
| BPE | 1.082 |

Summary:
- **The latent loss gives −0.030 BPB (~2.6%),** down from −0.089 at 0.6B.
- **It's worth about 1.15–1.4× effective compute.** The control's curve falls $0.226 / \log_2 5 \approx 0.097$ BPB per doubling of data, so 0.030 BPB ≈ $2^{0.31} \approx 1.24\times$ compute, or ~1.16× net of the extra ~7% FLOPs from the EMA target encoder and extra chunks. The top of the range assumes the curve flattens after 3B.
- **The N6·T9 baseline looks under-tuned.** An outside small-scale reproduction ([bigwolfeman/hnet-research](https://github.com/bigwolfeman/hnet-research), unverified) has 1-stage H-Net ~0.05 BPB behind BPE, which matches our N5·T7 control rather than N6·T9.
- **Rough rating: 4/10 as a BPB result.** The setup itself looks new: a JEPA-style loss on learned dynamic chunks, plus word-aligned boundaries emerging from it.

Runs, in job-list format for `scripts/run_queue.sh`:

1. Second seed of the no-latent control at 3B (~1.5 h):
```
python -m latents.train configs/s0_hnet.yaml group=Final-3B name=f3_plain_N5_T7_s1 model.N=5 model.arch_layout=[m4,[T7],m4] train.lr=2e-3 train.seed=1
```
2. Compute-matched control: ~7% more bytes. Or plot BPB against FLOPs with the EMA forward counted; it's currently left out of `train_flops`.
```
python -m latents.train configs/s0_hnet.yaml group=Final-3B name=f3_plain_N5_T7_3p2B model.N=5 model.arch_layout=[m4,[T7],m4] train.lr=2e-3 train.total_bytes=3.2e9
```
3. 10B bytes, latent vs no-latent (~5 h each). This decides whether the BPB gain survives more data.
```
python -m latents.train configs/hc_best.yaml group=Scale-10B name=f10_D3_mlp_N5_T7_s0 model.N=5 model.arch_layout=[m4,[T7],m4] train.total_bytes=1e10 train.eval_every=4000
python -m latents.train configs/s0_hnet.yaml group=Scale-10B name=f10_plain_N5_T7_s0 model.N=5 model.arch_layout=[m4,[T7],m4] train.lr=2e-3 train.total_bytes=1e10 train.eval_every=4000
```
4. Decay the latent weight to 0 during the cooldown (needs a `beta_tf` schedule in `latents/latent.py`). If the latent loss mainly speeds up early training, this keeps the gain without the late cost.

Benchmarks (eval-only, on any checkpoint):
- **Out-of-distribution BPB:**
  - code
  - Chinese: no spaces, so it tests whether word-aligned chunks are a whitespace trick
  - FineWeb-2 languages
  - Wikipedia
- **Robustness:** BPB on perturbed text, and perturbed HellaSwag with the BLT/H-Net perturbations (AntSpeak, Drop, RandomCase, Repeat, UpperCase).
- **Zero-shot tasks with signal at ~30M params:** BLiMP, LAMBADA, ARC-Easy, SciQ, PIQA, HellaSwag. Score them by BPB of the answer continuation, which is less noisy than accuracy at this size (Heineman et al., arXiv 2508.13144).
- **Skip:** MMLU, WinoGrande, ARC-Challenge, OpenBookQA and CUTE. They sit at chance at this size.
- **Chunk quality:**
  - boundary F1 against gold segmentations: SIGMORPHON morphemes, Chinese word segmentation (PKU/MSR)
  - linear probes on the chunk latents
