# Next steps

Updated Nov 10, 2026. Everything below is implemented; nothing has run on a GPU yet.

## Goal

The trunk (encoder, router, and a predictor at every level) learns only from latent prediction: each level predicts the latent of its next chunk. Bytes train only a readout decoder, which can't send gradient back into the trunk. Levels nest as deep as the data supports, and whether to go up another level is itself a routing decision.

Two losses stay because they constrain the model rather than carry information about the text: the ratio loss (how many chunks, i.e. compute) and an anti-collapse regularizer.

## Running it

We run on one pod with **2× H100 80GB**. The job runner uses whatever GPUs it sees, so 2, 4 or 8 of any 80GB card work.

Lambda, 2× H100 SXM, image **GPU Base 24.04** (any Ubuntu 22.04/24.04 image works). Set up as the `ubuntu` user, inside tmux, so an SSH drop doesn't kill the queue:

```bash
git clone https://github.com/ahilanks/latents && cd latents
```

```bash
cp /path/to/your/.env .env
```

```bash
tmux new -s ladder
```

```bash
bash setup.sh
```

```bash
source .venv/bin/activate
```

```bash
python -m pytest -q
```

```bash
bash scripts/ladder/run_all.sh
```

- `setup.sh` creates `.venv` with Python 3.12 (the pinned kernel wheels are 3.12 builds) and installs torch 2.8 + CUDA 12.8 into it.
- `python -m pytest -q` runs the 34 CPU tests (~30 s, kernel stubs, no GPU needed).
- `run_all.sh` runs preflight, phases 1–6 and the eval suite, and stops at the first failed gate.

On 2 GPUs the ladder uses both, and the side track runs after it (whether or not the ladder stopped at a gate). `touch logs/skip_side` before the ladder ends to skip it:

```bash
bash scripts/ladder/run_all.sh 2>&1 | tee -a logs/run_all.out; [ -e logs/skip_side ] || python scripts/run_jobs.py scripts/ladder/side.txt 2>&1 | tee -a logs/side.out
```

On 4 GPUs, give the side track its own GPU so it overlaps the ladder:

```bash
bash scripts/ladder/run_all.sh --gpus 0,1,2
```

```bash
python scripts/run_jobs.py scripts/ladder/side.txt --gpus 3
```

Pieces:
- `scripts/ladder.py` holds every rung as code: config, reference, early-kill rule and pass checks. `check <ID>` runs a gate, `select P2|P3|P4` picks a phase winner, `report` prints the table, `jobs` regenerates the job files.
- `scripts/run_jobs.py` runs a job file with one job per free GPU, or several per job via torchrun.
  - Gates are asynchronous: later rungs start speculatively while a gate waits on its runs.
  - If a gate fails, everything launched after it is killed and the queue stops.
  - A soft gate only skips the rest of its block.
  - It sends a Pushover notification for every gate, crash and stop if `PUSHOVER_*` is set in `.env`.
- `scripts/preflight.py` runs first. For every config, at real size, it checks:
  - a full forward/backward, its memory, and that all losses are finite
  - the gradient wiring (for example, with `detach_trunk` the byte loss reaches neither the encoder nor the main net)
  - a 4-step training run, and a strict checkpoint reload

  That takes about 10 minutes on 4 GPUs and catches crashes before they cost hours.
- **Early kill** (`train.kill_ref`), only after 80% of training and never on a gap under 0.05 BPB (run-to-run noise, see below):
  - neutral rungs are killed if they are that much worse than the rung below at the same step
  - rungs that remove a signal are killed if they fall that far below the floor control
  - a run judged hopeless can also be stopped by hand: write the reason to `runs/<name>/KILL`

### Compute plan

| Pod | $/hr | Ladder + side track | Cost |
|---|---|---|---|
| **2× H100 80GB (chosen)** | ~$8 | ~8.5–9.5 h, then ~3.5 h (side track runs after the ladder) | ~$100–110 |
| 4× H100 80GB | $16.36 | ~7–9 h | ~$115–150 |
| 8× A100 80GB | $22.32 | ~8.5 h | ~$190 |
| 4× A100 80GB | — | ~11 h | — |

Notes:
- **Measured vs guessed.** The A100 numbers come from measured run times in `logistics.md`. On the 2× H100 pod, the first 0.6B phase-1 run (`L-R0-s0`) ran at 0.24 s/step and took 9.9 min, against ~0.4 s/step and 15–16 min on the A100: about 1.7× faster. The 4× H100 row still assumes 1.3–1.6×. Rollout and nested rungs cost more per step, and their times are scaled from the A100 runs.
- **Assumptions.** Every gate passes, and the pod is stopped once the queue ends. Failed rungs or fallbacks add 10–20%.
- **Avoid 40GB cards.** Micro-batches are sized for 80GB.

## How we climb

- **One change per rung.** Each rung is compared to the rung right before it.
- **0.6B bytes, 2 seeds, compared paired by seed.** Runs take ~15 min. Latent runs at 0.6B differ by up to ~0.025 BPB between seeds, but recipe rankings have held across seeds.
- **A neutral change** (expected to cost nothing) passes if the paired Δ is at most +0.03 BPB (`TOL` in `scripts/ladder.py`) and the latent-vs-copy ratio is at most 1.10× the rung below.
  - **Why 0.03 and not 0.01:** on the H100s, identical same-seed reruns end ~0.015 BPB apart (R0-s0 1.2872 vs 1.2733) and same-seed runs spread ~0.02. So a paired 2-seed Δ has an sd of ~0.02, and a 0.01 tolerance was a coin flip.
  - Mid-training gaps are much noisier: ~0.06 at 45% of a run.
  - If a decisive rung (R6, the phase winners, the milestones) lands within ~0.02 of its threshold, add a third seed.
- **A costly change** (one that removes a signal) passes if the trunk still carries information (trunk info stays above 0.01) and the latent still beats the copy baseline. Log the cost.
- **If a rung fails,** try the alternative listed for it before moving up. Never stack a second change on a failing rung.
- **Milestones** get confirmed at 3B bytes.
- **Log every rung** in `logistics.md`.

## Metrics on every rung

All of these are in each run's `summary.json` and in W&B.
- **readout BPB:** `val/bpb` of the byte decoder.
- **floor BPB:** the same decoder with every trunk input zeroed, i.e. a small local byte model (`L-FLOOR`, `L-FLOOR3B`).
- **trunk info = floor − readout.** This is the main number: how much the latent trunk actually tells the decoder.
- **latent quality:**
  - `val/lat/ratio_tf_copy`: prediction L1 divided by the copy baseline, so below 1 means better than copying the last chunk
  - `val/lat/cos`
  - `val/lat/tgt_eff_rank`
- **document-ID shortcut detector:** `val/lat/tgt_within_frac`, the share of latent variance that is within sequences.
- **chunking:** `val/compression_L1/L0`, `val/edge_prec`, `val/space_recall`; per level, `val/L<s>/compression`.
- **rollout:** `val/lat/r<k>_l1` vs `val/lat/copy<k>_l1`, per horizon k (and `val/L<s>/...` per level).

## Phase 1: pure latent trunk, fixed boundaries

Start from the current best joint recipe and remove the byte signal one part of the stack at a time. Configs are in `configs/ladder/`.

| Rung | One change | Pass if | Config |
|---|---|---|---|
| R0 | Reference: the `hcD3` recipe (EMA 0.99, MLP head, N=5, main T7, joint) re-run on the new code | within 0.01 of hcD3/hcD4 (1.2733 / 1.2970); catches anything the refactor changed | `r0_ref` |
| R1 | Decoder gets its own byte embedding (zero-init, so it starts exactly at R0). Also run the floor. | neutral | `r1_dec_emb`, `floor` |
| R2 | Decoder reads the predicted latent $\hat c_{j+1}$ instead of the raw main-net output | trunk info > 0.01, beats copy (was +0.031 BPB on the old recipe) | `r2_decode_pred` |
| R3 | No byte gradient into the main net | as R2 (was +0.008) | `r3_detach_main` |
| R4 | Fixed boundaries, one every 4 bytes, instead of the router | Δ ≤ +0.03 vs R3 (fixed chunks tied learned ones at 0.6B in S0) | `r4_fixed_chunks` |
| R5 | Within-sequence variance floor on the chunk latents; byte gradient still reaches the encoder | neutral | `r5_within_reg` |
| R6 | No byte gradient into the encoder (detached residual). The decoder is now a pure readout. | trunk info > 0.01, beats copy, within-frac and eff. rank ≥ half of R5's | `r6_detach_trunk` |

**Milestone 1:** R6 at 3B, 2 seeds, against the floor at 3B. The whole trunk now learns only from the latent loss, with fixed boundaries.

Notes:
- **R1** sets up R6. Once the encoder stops being trained for bytes, the decoder needs its own view of the current bytes.
- **R4** uses fixed chunks so the router problem (design notes) can wait until phase 2. With no router, the residual is the only byte path left into the encoder, so R6 is a single detach.
- **R5's floor calibrates itself.** It is set to half the within-sequence std measured over the first 10% of training. So it sits inactive while the byte loss shapes the encoder, and only bites if the latent-only encoder starts collapsing within sequences. SIGReg (`latent.reg=sigreg`) and VICReg (`vicreg`) are available too.
- **R6 is the go/no-go rung.** The expected failure is the document-ID shortcut. If R6 fails, run `scripts/ladder/phase1_fallback.txt`: CPC-style negatives from the same sequence instead of the variance floor (R6b).

## Phase 2: learn the boundaries from the latent signal only

Each rung replaces R4's fixed chunks with N=4 learned boundaries (R4's chunk size) and is compared to R6. `select P2` keeps the best passing one, writes `configs/ladder/p2_best.yaml`, and runs it at 3B (**milestone 2**).

From here on, the decoder path also detaches the boundary probability $p$. The dechunk smoothing and the STE both use $p$, so without the detach the byte loss would leak back into the encoder through the router. That keeps R6's guarantee intact; it isn't a separate change.

| Rung | One change | Pass if | Config |
|---|---|---|---|
| R7 | Option D: router with $q, k$ frozen at identity, so cuts fall where the latent-trained encoder changes direction; the ratio loss sets the count | compression within ±25% of 0.25; readout within R6 + 0.01 | `r7_identity_router` |
| R8 | Option A: trainable router, with gradient from an STE on the predictor input | compression ok; readout ≤ R7, or edge precision ≥ 1.05× R7 | `r8_ste_router` |
| R9 | Option C: cut where a small byte-level latent predictor is surprised (running threshold steered to 1/N, fixed chunks during a 10% warmup) | as R8 | `r9_surprise_router` |

Option E (REINFORCE on sampled boundaries, reward = −latent loss) is in `phase2_fallback.txt`, for when none of the three passes.

## Phase 3: longer horizons, one level

| Rung | One change | Pass if | Config |
|---|---|---|---|
| R10 | 2-step rollout loss with predictor-only gradients (the S2c recipe) | 2-step error below its copy baseline; readout within the P2 winner + 0.01 | `r10_rollout2` |
| R11 | k-step rollout, k = 4, then 8 | error at step k below the copy baseline at k; readout within the shorter horizon + 0.01 | `r11a_rollout4`, `r11b_rollout8` |

The k-step rollout is exact. Every anchor gets its own chain of imagined chunks, and each chain token attends to the real context up to its anchor plus its own earlier chain tokens.

The gates here are soft: if a horizon fails, the longer ones are skipped and `select P3` takes the longest that passed. If none passed, it falls back to the P2 winner. Watch the seeds here: S2c split into a good and a bad outcome across seeds before.

## Phase 4: a second level

| Rung | One change | Pass if | Config |
|---|---|---|---|
| R12 | Add level 2 on top: encoder (m2) + predictor (T3) over level-1 chunks, fixed chunks of 3, its own latent loss. Nothing flows back down. | level-2 latent beats its copy baseline; level 1 unchanged (|Δ| ≤ 0.01) | `r12_level2_passenger` |
| R13 | Top-down: level-2 predictions condition the level-1 predictor | level-1 latent ratio and readout no worse than R12 | `r13_topdown` |
| R13b | Level-2 losses may send gradient into the level-1 chunks | neutral | `r13b_input_grad` |
| R14 | Learned level-2 boundaries (ratio target 3) | neutral; level-2 compression within ±30% of 1/3 | `r14_level2_router` |
| R15 | Rollout at level 2 (2 steps) | as R10 | `r15_level2_rollout` |

`select P4` takes R15 if it passed, else R14, and runs it at 3B (**milestone 3**).

Notes:
- **R12** is a passenger check: level 2 learns, but can't help or hurt level 1 yet.
- **R13** tests whether level 2 helps. The top-down path is trained only by level 1's latent loss, so this is still latent signals only.
- **Top-down is built exactly like H-Net's decoder path:** a dechunk with EMA smoothing by $p$, times STE(selected prob), then a zero-initialised projection. So from level 2 up, the router gets its gradient from the latent loss of the level below for free. The router problem from phase 2 only exists at level 1.

## Phase 5: nest indefinitely

| Rung | One change | Pass if | Config |
|---|---|---|---|
| R16 | Tie weights across levels: one block reused at every level (the level-1 predictor everywhere), plus a level embedding | within the P4 winner + 0.01 | `r16_tie` |
| R17 | Add level 3 using the shared block | level-2 latent ratio no worse than R16 | `r17_level3` |
| R18 | Choose depth by a rule: keep going up while the expected top-level length is above 128 chunks (cap 6) | |Δ| ≤ 0.01 vs R17 | `r18_depth_rule` |
| R19 | Learned depth: above level 2, a compute penalty on the boundary probability replaces the fixed ratio target, so a level can send (almost) nothing up where going up doesn't help | within R18 + 0.01 at ≤ 1.05× its FLOPs | `r19_compute_penalty` |

Notes:
- **R18's rule uses expected lengths:** window length divided by the chunk sizes. So the depth decision never depends on the window's content, and the model stays causal; a rule on actual lengths would leak a bit about the future. With 8KB windows the rule gives 4 levels (2048 → 683 → 228 → 76 chunks), so R18 also tests whether a fourth level helps.
- **How R19 turns "go up?" into a routing decision:** the next level's router already picks which chunks go up. With a compute penalty instead of a fixed target, it can pick (almost) none in a region, so depth varies by region. The payoff for going up is still a latent signal: the router's gradient comes from the latent loss of the level below, through the top-down STE.
- **Practical limit:** 8KB windows. Going deeper needs longer windows.

## Phase 6: use it

- **Latent chain of thought** (`latents/tasks.py`, `scripts/latent_cot.py`, `scripts/ladder/phase6.txt`):
  - **Fine-tuning.** Fine-tune the phase-3 winner (and the R0 joint model, for comparison) on synthetic multi-step arithmetic with written-out chains of thought.
  - **Three ways to answer:**
    - directly
    - with a byte-level chain of thought
    - with k latent steps: the predicted latents are fed back into the main net without decoding, for k = 2, 4, 8, 16, and the number of chunks the real chain of thought would have taken
  - **Reported:** accuracy by number of steps, sequential steps and FLOPs.
  - **Savings per chunk** on the 1-level model (~3.45 bytes per chunk): a latent step costs ~62 MFLOP vs ~101 MFLOP to spell the chunk out, i.e. 1.6× fewer FLOPs and ~3.45× fewer sequential steps.
  - 1-level models only for now.
- **Benchmarks** at each milestone (`scripts/eval_suite.py`, `scripts/ladder/evals.txt`). The list is in the side track.

## Design notes

### Collapse vs shortcut

Stop-grad plus EMA already prevents collapse in our runs (target effective rank 175–208 of 256). SIGReg (LeJEPA) or VICReg would make that sturdier.

The risk without the byte loss is a **shortcut** instead:
- **What it looks like.** The encoder learns a code that is nearly constant within a document but differs across documents (a topic or document ID).
- **Why the regularizers miss it.** SIGReg and VICReg only constrain the spread *across the batch*, so this code passes them. Yet within a document it's trivially predictable: just copy the last chunk.
- **Detect it with** `val/lat/tgt_within_frac` and the copy-baseline ratio.
- **Fix 1 (R5):** apply the regularizer *within* sequences (each sequence centred first).
- **Fix 2 (R6b):** CPC-style InfoNCE with negatives drawn from the same sequence. The model then has to pick *which* chunk comes next, not just which document it's in.

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
- **A. STE on the predictor input (R8).** Feed $c_j \cdot \mathrm{STE}(p_{b_j})$ to the main net. The forward pass is unchanged, and the backward pass gives $\partial \mathcal{L}_{\text{lat}} / \partial p_{b_j} = \langle \partial \mathcal{L}_{\text{lat}} / \partial c'_j,\, c_j \rangle$, i.e. "does this chunk help prediction?" Limit: only existing boundaries get a signal.
- **B. Soft chunk inputs.** Build $c_j$ as a $p$-weighted average of the byte states in the chunk, so every byte's $p$ gets a gradient. Context side only: if the targets are soft too, the router can pick easy targets. Not implemented, since A, C and D cover the space.
- **C. Prediction-error boundaries (R9).** Cut where a small byte-level latent predictor is surprised. This is BLT's entropy patching moved into latent space, and it matches event-segmentation theory from cognitive science. No gradient through discrete choices at all.
- **D. Identity router (R7).** Freeze $q$ and $k$ at identity and let the latent-trained encoder decide. The ratio loss sends gradient into $h$ through $p$.
- **E. REINFORCE on sampled boundaries.** General but noisy. Last resort.

Shared danger: the router can game the latent loss by cutting where prediction is easy, e.g. right after very predictable spans. Readout BPB, edge precision and the copy baseline keep it honest.

### The decoder as a readout

With all of its inputs detached, the decoder is a probe. Readout BPB then measures how much of the text the latent trunk keeps, the analogue of linear-probe accuracy in vision JEPA.

It needs its own byte embedding (R1). Under latent-only training, only the encoder states at boundary bytes ever get a gradient, so the per-byte residual stops being a reliable view of the current bytes.

## Side track: is the current joint result real?

`scripts/ladder/side.txt`. These runs don't block the ladder.

| @3B bytes | val BPB |
|---|---|
| H-Net N6·T9 | 1.275 |
| H-Net N5·T7, no latent loss (1 seed) | 1.136 |
| + latent loss (2 seeds) | 1.107 |
| BPE | 1.082 |

Summary:
- **The latent loss gives −0.030 BPB (~2.6%),** down from −0.089 at 0.6B.
- **It's worth about 1.15–1.4× effective compute.** The control's curve falls $0.226 / \log_2 5 \approx 0.097$ BPB per doubling of data, so 0.030 BPB ≈ $2^{0.31} \approx 1.24\times$ compute, or ~1.16× net of the extra ~7% FLOPs. The top of the range assumes the curve flattens after 3B.
  - FLOPs are now counted exactly per batch, including the EMA forward, rollout passes and towers.
- **The N6·T9 baseline looks under-tuned.** An outside small-scale reproduction ([bigwolfeman/hnet-research](https://github.com/bigwolfeman/hnet-research), unverified) has 1-stage H-Net ~0.05 BPB behind BPE, which matches our N5·T7 control rather than N6·T9.
- **Rough rating: 4/10 as a BPB result.** The setup itself looks new: a JEPA-style loss on learned dynamic chunks, plus word-aligned boundaries emerging from it.

Runs:
1. Second seed of the no-latent control at 3B.
2. Compute-matched control: ~7% more bytes.
3. 10B bytes, latent vs no-latent. This decides whether the BPB gain survives more data.
4. The latent weight decayed to 0 during the cooldown (`latent.beta_schedule=cooldown`), 2 seeds at 0.6B.

Benchmarks (`scripts/eval_suite.py fetch`, then `run runs/<name>`; works for every model type):
- **Out-of-distribution BPB:**
  - code
  - Chinese and English Wikipedia
  - FineWeb-2 (Russian, Chinese, Hindi, Arabic, German)
- **Robustness:** BPB on perturbed FineWeb-Edu val text (antspeak, drop, randomcase, repeat, uppercase).
- **Zero-shot tasks:**
  - HellaSwag, PIQA, ARC-Easy, SciQ: accuracy, per-byte accuracy, and BPB of the correct answer
  - LAMBADA: target BPB and greedy exact match
  - BLiMP
- **Chunk quality:**
  - morpheme-boundary F1 against SIGMORPHON 2022
  - linear probes on the chunk latents (the current word, and the next word from the predicted latent)
- **Skip:** MMLU, WinoGrande, ARC-Challenge, OpenBookQA and CUTE. They sit at chance at this size.
