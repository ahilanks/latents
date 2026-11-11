# latents — H-Net → JEPA-style latent prediction over dynamic chunks

**Goal.** Show results for a "latent-style" H-Net: the main network acts as a predictor over the
H-Net's dynamic chunks and is trained V-JEPA 2-AC style. It predicts the next chunk's representation
from an EMA target encoder, with an L1 teacher-forcing loss plus a 2-step rollout loss:

```
L_TF = Σ_j ‖P(c_≤j) − sg(c̄_{j+1})‖₁        L_R = ‖P(c_≤j, ĉ_{j+1}) − sg(c̄_{j+2})‖₁,  ĉ_{j+1} = P(c_≤j)
```

## North star
- **Target architecture: the Recursive H-Net** ([design artifact](https://claude.ai/artifact/Bq16b6fRViYnRtkDgWeM4w)).
  - One shared encoder, chunker and decoder, looped across levels.
  - An up/predict depth router with halting weights.
  - Think/emit latent reasoning and a halt router.
  - Latent-only training, with the decoder on stop-gradient.
  - Long horizons and longer, temporally abstracted rollouts come from hierarchy, i.e. nesting.
- **Guiding principles:** section 1, "Forming a Rich Internal Dynamics Model", of [this write-up](https://ahilanks.github.io/writings/p/782e42e685f06f174968ab00f85ff192.html).
  - Predict in latent space and keep only the structure needed to predict consequences.
  - Prevent collapse without negatives: a stop-gradient EMA target, plus a regularizer such as SIGReg or VICReg.
  - No regularizer that ties back to the raw input (no reconstruction).
- **Rules that follow from these:**
  - Only non-contrastive objectives.
  - Nothing hard-coded (e.g. no task-aligned chunking).
  - The decoder gets its own loss, but its gradient never reaches the latents, encoder or trunk.
  - Train teacher-forced on real encodings only. The model "dreams" in its own rollouts only at inference.

## Ground rules (from the project brief)
1. First **reproduce H-Net at small scale** and match the best published small-scale results.
2. Then move toward the final setup **one small change at a time**. Each change is trained and
   compared against the previous step, and kept only if it helps (or its cost is understood and logged).
3. Log everything that matters, but not more than that:
   - W&B project `hnet-latents` (team entity `ahilan-uc-berkeley-electrical-engineering-computer-sciences`)
   - HF model repo [`a5ilank/hnet-latents`](https://huggingface.co/a5ilank/hnet-latents), one folder per run: `runs/<name>/`
   - [`logistics.md`](logistics.md): a timestamped log of what was tried and what worked or didn't
4. Any new dependency goes into `setup.sh`, which is what a fresh pod runs.

## Plan
| Phase | Change vs previous | Pass criterion |
|---|---|---|
| **S0** reproduce | Compute-matched (~2.3e17 FLOPs, 3B bytes of FineWeb-Edu) BPE Transformer vs isotropic byte Transformer vs H-Net (pool) vs H-Net 1-stage (DC) | Paper ordering: H-Net ≈ BPE < pool < byte-isotropic in val BPB. Learned compression ≈ 1/N. Gaps comparable to main-horse's small-scale sweep / paper Table 2 |
| **S1** latent aux | + teacher-forced latent loss `L_TF` (main net → next-chunk rep), first with stop-grad online targets, then EMA targets | BPB not worse than S0 H-Net; latent loss ↓; no collapse (target std / eff. rank) |
| **S2** rollout | + 2-step rollout loss `L_R` (flex-attention masked second pass) | rollout error ↓ without hurting BPB/`L_TF` |
| **S3** cut top half | main net trained by latent losses only (AR gradient stopped at main net); decoder reads the predicted next-chunk latent | quantify the cost: BPB and latent metrics vs joint S1/S2 + zero-context control |

## Findings so far (0.6B bytes, val BPB on FineWeb-Edu; seed noise ≈ ±0.003, latent runs with rollout ≈ ±0.013)
- **Pipeline check:** official `hnet_1stage_L` on our val = 0.756 (paper 0.755).
- **S0 at small scale:** BPE 1.191 < byte Transformer 1.347 < H-Net 1.372 ≈ pool 1.375. The main net adds only 0.038 BPB (zero-main control 1.411).
- **S1 ✅ teacher-forced latent loss** (next-chunk L1 vs stop-grad/EMA(0.99) targets): 1.343–1.350, about −0.025 BPB, robust over settings. A compression-matched control rules out chunk size as the cause.
  It can also induce word-aligned chunks; word alignment tracks latent quality, not BPB.
- **S2 rollout:** with full gradients it hurts. With predictor-only gradients (V-JEPA 2-AC style) it is BPB-neutral on average, but it bifurcates across seeds (best: cos 0.73, word-aligned chunks).
- **S3 latent-style model** (predictor trained only in latent space, decoder reads predicted latents): 1.404 vs joint 1.365.
  Most of that cost (0.031) is the latent bottleneck; dropping AR gradients costs only 0.008.
- **Hillclimb (Nov 8):** EMA(0.99) targets + **MLP predictor head** + more, smaller chunks (N=5) + shallower main net (T7).
  **1.285** (2-seed mean) at the byte Transformer's FLOPs: −0.088 vs H-Net and −0.062 vs the byte Transformer.
  Every step holds when paired by seed. Rejected: LN targets, extra horizons, β≥1 with a linear head.
  The no-latent control with the same architecture gets 1.362, so the gain is from the latent loss.
- **3B bytes (2 seeds):** final **1.107 ± 0.001** vs H-Net 1.275, byte Transformer 1.232, BPE 1.082 (closes ~87% of the H-Net→BPE gap).
  ⚠️ The no-latent control with the same architecture gets 1.136 @3B. So at 3B, most of the gain is the hillclimbed architecture (finer chunks N=5, shallower main: −0.139).
  The latent loss adds −0.029 (at +11–18% FLOPs from even finer chunks). At 0.6B the split was reversed (−0.010 arch, −0.089 latent). Causality verified (`tests/test_causality2.py`).
Details and per-run notes: [`logistics.md`](logistics.md).

## Layout
```
latents/            prepare_data.py (FineWeb-Edu -> byte + GPT-2 streams), models.py (official H-Net wrapper, forward
                    options for the ladder), train.py (trainer: 1 GPU / torchrun / CPU, early kill, exact FLOPs),
                    latent.py (JEPA objectives, regularisers, CPC, levels), rollout.py (exact k-step rollout),
                    routing.py (surprise / sampled routers), nested.py (levels >= 2), flops.py, tasks.py (phase-6 task)
configs/            base.yaml + one yaml per experiment (inheritance via `base:`); ladder/ = one config per rung,
                    side/ = side-track runs, task/ = phase-6 fine-tunes
scripts/            ladder.py (rungs, gates, report, job files), run_jobs.py (multi-GPU queue with gates),
                    preflight.py (GPU smoke test), eval_suite.py (OOD/perturbed BPB, zero-shot, chunk probes),
                    latent_cot.py (phase 6), ladder/*.txt job files + run_all.sh, older phase job lists
tests/              cpu/ (34 CPU tests with pure-PyTorch kernel stubs), GPU scripts test_*.py
third_party/hnet    official goombalab/hnet @ 3673fe1 (installed by setup.sh)
```

## Usage
```bash
bash setup.sh                                   # deps, pinned kernels, official H-Net, data (SKIP_DATA=1 to skip)
python -m pytest -q                             # CPU tests (tests/cpu)
bash scripts/ladder/run_all.sh                  # the whole ladder in NEXT_STEPS.md, gated, on all visible GPUs
python scripts/ladder.py report                 # rung table with paired deltas and pass/fail
python -m latents.train configs/s0_hnet.yaml    # a single run; any key can be overridden: train.lr=1e-3 name=foo
```
