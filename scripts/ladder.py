"""The ladder in NEXT_STEPS.md as code: rung specs, pass checks, phase-winner selection, report, job files.

usage:
  python scripts/ladder.py check <ID>     exit 0 pass / 1 fail / 2 results not complete yet
  python scripts/ladder.py select <ID>    pick a phase winner (P2, P3, P4), write configs/ladder/<id>_best.yaml
  python scripts/ladder.py report         table of every rung with results
  python scripts/ladder.py jobs           (re)write scripts/ladder/*.txt job files for scripts/run_jobs.py

Runs are named L-<ID>-s<seed> (0.6B bytes) and live in runs/<name>/summary.json. Every comparison is paired
by seed: Δ = mean over seeds of (rung - reference). Reference "@P2" means the rung that won phase 2.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs" / "ladder_state.json"
SEEDS = (0, 1)
OLD_R0 = {0: 1.2733, 1: 1.2970}  # hcD3 / hcD4: same recipe on the previous code
TRAIN = "python -m latents.train"


# ----------------------------------------------------------------------------- results
def state():
    return json.load(open(STATE)) if STATE.exists() else {}


def resolve(rid):
    if rid.startswith("@"):
        st = state()
        if rid[1:] not in st:
            raise KeyError(f"{rid}: phase winner not selected yet")
        return st[rid[1:]]
    return rid


def run_name(rid, seed):
    return f"L-{rid}-s{seed}"


def summary(rid, seed):
    p = ROOT / "runs" / run_name(rid, seed) / "summary.json"
    return json.load(open(p)) if p.exists() else None


class NotReady(Exception):
    pass


def results(rid, key="val/bpb", seeds=None):
    """{seed: value}; raises NotReady if a run has not finished, returns None values for killed runs."""
    rid = resolve(rid)
    out = {}
    for s in seeds or RUNGS.get(rid, {}).get("seeds", SEEDS):
        sm = summary(rid, s)
        if sm is None:
            raise NotReady(f"{run_name(rid, s)} not finished")
        out[s] = None if sm.get("killed") else sm.get(key)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def paired(rid, ref, key="val/bpb"):
    a, b = results(rid, key), results(ref, key)
    common = [s for s in a if s in b]
    if any(a[s] is None or b[s] is None for s in common) or not common:
        return None
    return sum(a[s] - b[s] for s in common) / len(common)


# ----------------------------------------------------------------------------- checks (rid -> (ok, msg))
def delta_le(ref, tol, key="val/bpb"):
    def f(rid):
        d = paired(rid, ref, key)
        return (d is not None and d <= tol), f"Δ{key.split('/')[-1]} vs {resolve(ref)} = {d if d is None else round(d, 4)} (<= {tol})"
    return f


def delta_abs_le(ref, tol, key="val/bpb"):
    def f(rid):
        d = paired(rid, ref, key)
        return (d is not None and abs(d) <= tol), f"|Δ{key.split('/')[-1]}| vs {resolve(ref)} = {d if d is None else round(d, 4)} (<= {tol})"
    return f


def near_const(vals, tol):
    def f(rid):
        r = results(rid)
        if any(v is None for v in r.values()):
            return False, "killed"
        d = mean(r[s] - vals[s] for s in r)
        return abs(d) <= tol, f"Δ vs previous code {d:.4f} (|.| <= {tol}); runs {r}"
    return f


def trunk_info_gt(floor, x):
    def f(rid):
        r, fl = results(rid), results(floor)
        fm, rm = mean(fl.values()), mean(r.values())
        if fm is None or rm is None:
            return False, "missing/killed"
        return fm - rm > x, f"trunk info = floor {fm:.4f} - readout {rm:.4f} = {fm - rm:.4f} (> {x})"
    return f


def lt(key_a, key_b, factor=1.0):
    def f(rid):
        a, b = mean(results(rid, key_a).values()), mean(results(rid, key_b).values())
        if a is None or b is None:
            return False, f"{key_a} or {key_b} missing"
        return a < factor * b, f"{key_a} {a:.4f} < {factor} x {key_b} {b:.4f}"
    return f


def beats_copy(level=1):
    k = "val/lat/ratio_tf_copy" if level == 1 else f"val/L{level}/ratio_tf_copy"
    def f(rid):
        v = mean(results(rid, k).values())
        return (v is not None and v < 1.0), f"{k} = {v if v is None else round(v, 4)} (< 1)"
    return f


def ratio_le(ref, factor, key="val/lat/ratio_tf_copy"):
    def f(rid):
        a, b = mean(results(rid, key).values()), mean(results(ref, key).values())
        if a is None or b is None:
            return False, f"{key} missing"
        return a <= factor * b, f"{key} {a:.4f} <= {factor} x {resolve(ref)} {b:.4f}"
    return f


def ge_frac(ref, frac, key):
    def f(rid):
        a, b = mean(results(rid, key).values()), mean(results(ref, key).values())
        if a is None or b is None:
            return False, f"{key} missing"
        return a >= frac * b, f"{key} {a:.4f} >= {frac} x {resolve(ref)} {b:.4f}"
    return f


def near(key, target, rel):
    def f(rid):
        v = mean(results(rid, key).values())
        return (v is not None and abs(v - target) <= rel * target), f"{key} = {v} (target {target:.3f} ± {rel:.0%})"
    return f


def any_of(*checks):
    def f(rid):
        res = [c(rid) for c in checks]
        return any(ok for ok, _ in res), " OR ".join(m for _, m in res)
    return f


def flops_le(ref, factor):
    return ratio_le(ref, factor, key="train_flops")


# ----------------------------------------------------------------------------- the ladder
# cfg: configs/ladder/<cfg>.yaml   kill: (reference rung, margin) for latents.train early kill
# checks: all must pass. 3B milestones use train.total_bytes=3e9.
B3 = "train.total_bytes=3e9 train.eval_every=2000"
RUNGS = {
    # phase 1: pure latent trunk, fixed boundaries
    "R0": dict(phase=1, cfg="r0_ref", desc="reference: hcD3 recipe on the new code",
               checks=[near_const(OLD_R0, 0.01)]),
    "R1": dict(phase=1, cfg="r1_dec_emb", ref="R0", kill=("R0", 0.03), desc="decoder gets its own byte embedding",
               checks=[delta_le("R0", 0.01), ratio_le("R0", 1.05)]),
    "FLOOR": dict(phase=1, cfg="floor", desc="control: decoder alone (byte embedding only)", checks=[]),
    "R2": dict(phase=1, cfg="r2_decode_pred", ref="R1", kill=("FLOOR", 0.01), desc="decoder reads the predicted latent",
               checks=[trunk_info_gt("FLOOR", 0.01), beats_copy()]),
    "R3": dict(phase=1, cfg="r3_detach_main", ref="R2", kill=("FLOOR", 0.01), desc="no byte gradient into the main net",
               checks=[trunk_info_gt("FLOOR", 0.01), beats_copy()]),
    "R4": dict(phase=1, cfg="r4_fixed_chunks", ref="R3", kill=("R3", 0.06), desc="fixed boundaries every 4 bytes",
               checks=[delta_le("R3", 0.03), trunk_info_gt("FLOOR", 0.01), beats_copy()]),
    "R5": dict(phase=1, cfg="r5_within_reg", ref="R4", kill=("R4", 0.03), desc="within-sequence variance floor",
               checks=[delta_le("R4", 0.01), ratio_le("R4", 1.05)]),
    "R6": dict(phase=1, cfg="r6_detach_trunk", ref="R5", kill=("FLOOR", 0.01), desc="no byte gradient into the encoder",
               checks=[trunk_info_gt("FLOOR", 0.01), beats_copy(),
                       ge_frac("R5", 0.5, "val/lat/tgt_within_frac"), ge_frac("R5", 0.5, "val/lat/tgt_eff_rank")]),
    "R6b": dict(phase=1, cfg="r6b_cpc", ref="R5", kill=("FLOOR", 0.01), desc="fallback for R6: CPC negatives instead of the variance floor",
                checks=[trunk_info_gt("FLOOR", 0.01), beats_copy(),
                        ge_frac("R5", 0.5, "val/lat/tgt_within_frac"), ge_frac("R5", 0.5, "val/lat/tgt_eff_rank")]),
    "FLOOR3B": dict(phase=1, cfg="floor", extra=B3, seeds=(0,), desc="floor control at 3B", checks=[]),
    "M1": dict(phase=1, cfg="r6_detach_trunk", extra=B3, gpus="half", desc="milestone 1: R6 at 3B",
               checks=[trunk_info_gt("FLOOR3B", 0.01), beats_copy()]),
    # phase 2: boundaries from the latent signal only (alternatives to R4's fixed chunks)
    "R7": dict(phase=2, cfg="r7_identity_router", ref="R6", kill=("R6", 0.06), desc="identity router (option D)",
               checks=[near("val/compression_L1/L0", 0.25, 0.25), delta_le("R6", 0.01)]),
    "R8": dict(phase=2, cfg="r8_ste_router", ref="R7", kill=("R6", 0.06), desc="trainable router, STE on predictor input (option A)",
               checks=[]),  # checks set below
    "R9": dict(phase=2, cfg="r9_surprise_router", ref="R7", kill=("R6", 0.06), desc="prediction-error boundaries (option C)",
               checks=[near("val/compression_L1/L0", 0.25, 0.25), delta_le("R7", 0.0)]),
    "RE": dict(phase=2, cfg="re_reinforce", ref="R7", kill=("R6", 0.06), desc="fallback: REINFORCE on boundaries (option E)",
               checks=[near("val/compression_L1/L0", 0.25, 0.25), delta_le("R6", 0.01)]),
    "M2": dict(phase=2, cfg="p2_best", extra=B3, gpus="half", desc="milestone 2: phase-2 winner at 3B",
               checks=[trunk_info_gt("FLOOR3B", 0.01), beats_copy()]),
    # phase 3: longer horizons
    "R10": dict(phase=3, cfg="r10_rollout2", ref="@P2", kill=("@P2", 0.03), desc="2-step rollout, predictor-only gradients",
                checks=[lt("val/lat/r2_l1", "val/lat/copy2_l1"), delta_le("@P2", 0.01)]),
    "R11a": dict(phase=3, cfg="r11a_rollout4", ref="R10", kill=("R10", 0.03), desc="4-step rollout",
                 checks=[lt("val/lat/r4_l1", "val/lat/copy4_l1"), delta_le("R10", 0.01)]),
    "R11b": dict(phase=3, cfg="r11b_rollout8", ref="R11a", kill=("R11a", 0.03), gpus="half", desc="8-step rollout",
                 checks=[lt("val/lat/r8_l1", "val/lat/copy8_l1"), delta_le("R11a", 0.01)]),
    # phase 4: a second level
    "R12": dict(phase=4, cfg="r12_level2_passenger", ref="@P3", kill=("@P3", 0.03), desc="level 2 as a passenger",
                checks=[beats_copy(2), delta_abs_le("@P3", 0.01), ratio_le("@P3", 1.05)]),
    "R13": dict(phase=4, cfg="r13_topdown", ref="R12", kill=("R12", 0.03), desc="top-down into the level-1 predictor",
                checks=[ratio_le("R12", 1.0), delta_le("R12", 0.0)]),
    "R13b": dict(phase=4, cfg="r13b_input_grad", ref="R13", kill=("R13", 0.03), desc="level-2 losses reach level-1 chunks",
                 checks=[delta_le("R13", 0.01)]),
    "R14": dict(phase=4, cfg="r14_level2_router", ref="R13b", kill=("R13b", 0.03), desc="learned level-2 boundaries",
                checks=[delta_le("R13b", 0.01), near("val/L2/compression", 1 / 3, 0.3)]),
    "R15": dict(phase=4, cfg="r15_level2_rollout", ref="R14", kill=("R14", 0.03), desc="rollout at level 2",
                checks=[lt("val/L2/r2_l1", "val/L2/copy2_l1"), delta_le("R14", 0.01)]),
    "M3": dict(phase=4, cfg="p4_best", extra=B3, gpus="half", desc="milestone 3: two levels at 3B",
               checks=[trunk_info_gt("FLOOR3B", 0.01), beats_copy(), beats_copy(2)]),
    # phase 5: nesting
    "R16": dict(phase=5, cfg="r16_tie", ref="@P4", kill=("@P4", 0.03), desc="one block shared by every level",
                checks=[delta_le("@P4", 0.01)]),
    "R17": dict(phase=5, cfg="r17_level3", ref="R16", kill=("R16", 0.03), desc="third level",
                checks=[ratio_le("R16", 1.0, key="val/L2/ratio_tf_copy")]),
    "R18": dict(phase=5, cfg="r18_depth_rule", ref="R17", kill=("R17", 0.03), desc="depth by a fixed length rule",
                checks=[delta_abs_le("R17", 0.01)]),
    "R19": dict(phase=5, cfg="r19_compute_penalty", ref="R18", kill=("R18", 0.03), desc="learned depth via a compute penalty",
                checks=[delta_le("R18", 0.01), flops_le("R18", 1.05)]),
}
# R8: better than R7 on readout BPB, or on edge precision at no BPB cost
RUNGS["R8"]["checks"] = [near("val/compression_L1/L0", 0.25, 0.25),
                         any_of(delta_le("R7", 0.0), ge_frac("R7", 1.05, "val/edge_prec"))]
RUNGS["R9"]["checks"] = [near("val/compression_L1/L0", 0.25, 0.25),
                         any_of(delta_le("R7", 0.0), ge_frac("R7", 1.05, "val/edge_prec"))]

SELECT = {
    # phase winner: candidates in preference order; "min" = lowest mean val BPB among passing
    "P2": dict(cands=["R7", "R8", "R9", "RE"], how="min", fallback=None),
    "P3": dict(cands=["R11b", "R11a", "R10"], how="first", fallback="@P2"),
    "P4": dict(cands=["R15", "R14"], how="first", fallback=None),
}


def check(rid):
    spec = RUNGS[rid]
    try:
        res = [c(rid) for c in spec["checks"]]
    except NotReady as e:
        return None, str(e)
    killed = [s for s in spec.get("seeds", SEEDS) if (summary(rid, s) or {}).get("killed")]
    if killed:
        return False, f"killed seeds {killed}: " + "; ".join((summary(rid, s) or {}).get("killed", "") for s in killed)
    ok = all(o for o, _ in res)
    return ok, "; ".join(("PASS " if o else "FAIL ") + m for o, m in res) or "no checks"


def config_path(rid):
    return f"ladder/{RUNGS[rid]['cfg']}.yaml"


def select(pid):
    sp = SELECT[pid]
    passing = []
    for rid in sp["cands"]:
        if not all(summary(rid, s) for s in RUNGS[rid].get("seeds", SEEDS)):
            continue
        ok, msg = check(rid)
        print(f"{rid}: {'pass' if ok else 'fail'} - {msg}")
        if ok:
            passing.append(rid)
    if not passing and sp["fallback"] is None:
        return None
    if passing:
        win = min(passing, key=lambda r: mean(results(r).values())) if sp["how"] == "min" else passing[0]
        base = config_path(win)
    else:
        win = resolve(sp["fallback"])
        base = (ROOT / "configs" / "ladder" / f"{sp['fallback'][1:].lower()}_best.yaml").relative_to(ROOT / "configs").as_posix()
    st = state()
    st[pid] = win
    STATE.parent.mkdir(exist_ok=True)
    json.dump(st, open(STATE, "w"), indent=1)
    out = ROOT / "configs" / "ladder" / f"{pid.lower()}_best.yaml"
    out.write_text(f"# written by scripts/ladder.py select {pid}: winner {win}\nbase: {base}\n")
    print(f"{pid} winner: {win} -> {out.relative_to(ROOT)}")
    return win


def report():
    print("| rung | what | seeds val BPB | mean | Δ vs ref | trunk info | tf/copy | within | comp | edge | check |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for rid, spec in RUNGS.items():
        seeds = spec.get("seeds", SEEDS)
        sm = {s: summary(rid, s) for s in seeds}
        if not any(sm.values()):
            continue
        g = lambda k: mean([(v or {}).get(k) for v in sm.values()])  # noqa: E731
        vals = " / ".join(f"{(v or {}).get('val/bpb', float('nan')):.4f}" + ("k" if (v or {}).get("killed") else "") for v in sm.values())
        try:
            d = paired(rid, spec["ref"]) if spec.get("ref") else None
        except (NotReady, KeyError):
            d = None
        try:
            fl = mean(results("FLOOR3B" if spec.get("extra") else "FLOOR").values())
        except NotReady:
            fl = None
        ok, msg = check(rid) if spec["checks"] else (None, "")
        fmt = lambda x, n=4: "–" if x is None else f"{x:.{n}f}"  # noqa: E731
        print(f"| {rid} | {spec['desc']} | {vals} | {fmt(g('val/bpb'))} | {fmt(d)} | "
              f"{fmt(None if fl is None or g('val/bpb') is None else fl - g('val/bpb'))} | {fmt(g('val/lat/ratio_tf_copy'), 3)} | "
              f"{fmt(g('val/lat/tgt_within_frac'), 2)} | {fmt(g('val/compression_L1/L0'), 3)} | {fmt(g('val/edge_prec'), 2)} | "
              f"{'' if ok is None else ('pass' if ok else 'FAIL')} |")


# ----------------------------------------------------------------------------- job files
def job(rid, seed):
    spec = RUNGS[rid]
    args = [TRAIN, f"configs/{config_path(rid)}", f"name={run_name(rid, seed)}", f"train.seed={seed}",
            f"group=ladder-P{spec['phase']}"]
    if spec.get("extra"):
        args.append(spec["extra"])
    if spec.get("kill"):
        ref, margin = spec["kill"]
        ref_name = run_name(ref, seed) if not ref.startswith("@") else f"$(python scripts/ladder.py name {ref} {seed})"
        args += [f"train.kill_ref={ref_name}", f"train.kill_margin={margin}"]
    gp = spec.get("gpus")
    return (f"[gpus={gp}] " if gp else "") + " ".join(args)


def block(rid):
    return [job(rid, s) for s in RUNGS[rid].get("seeds", SEEDS)]


PHASES = {
    "phase1": ["# Phase 1: pure latent trunk with fixed boundaries (NEXT_STEPS.md). ~16 0.6B runs + 3 at 3B.",
               *block("R0"), "gate R0", *block("R1"), *block("FLOOR"), "gate R1", *block("FLOOR3B"), *block("R2"),
               "gate R2", *block("R3"), "gate R3", *block("R4"), "gate R4", *block("R5"), "gate R5",
               *block("R6"), "gate R6", "barrier", *block("M1"), "barrier", "gate M1"],
    "phase1_fallback": ["# Only if gate R6 failed: CPC negatives instead of the variance floor, then milestone 1 on R6b.",
                        *block("R6b"), "gate R6b", "barrier",
                        *[j.replace("r6_detach_trunk", "r6b_cpc") for j in block("M1")], "barrier", "gate M1"],
    "phase2": ["# Phase 2: learn the boundaries from the latent signal only. 6 runs, then the winner at 3B.",
               *block("R7"), *block("R8"), *block("R9"), "barrier", "select P2", *block("M2"), "barrier", "gate M2"],
    "phase2_fallback": ["# Only if select P2 found no passing router: REINFORCE (option E).",
                        *block("RE"), "barrier", "select P2", *block("M2"), "barrier", "gate M2"],
    "phase3": ["# Phase 3: longer horizons. Soft gates: if a horizon fails, the longer ones are skipped.",
               *block("R10"), "gate R10 soft", *block("R11a"), "gate R11a soft", *block("R11b"), "barrier", "select P3"],
    "phase4": ["# Phase 4: a second level.",
               *block("R12"), "gate R12", *block("R13"), "gate R13", *block("R13b"), "gate R13b",
               *block("R14"), "gate R14", *block("R15"), "barrier", "select P4", *block("M3"), "barrier", "gate M3"],
    "phase5": ["# Phase 5: nest indefinitely.",
               *block("R16"), "gate R16", *block("R17"), "gate R17", *block("R18"), "gate R18 soft", *block("R19"), "barrier"],
}


SIDE = [
    "# Side track (independent of the ladder): is the current joint result real? Fill idle GPUs with these.",
    f"{TRAIN} configs/side/plain_N5_T7.yaml name=side-f3-plain-s1 train.seed=1 {B3}",
    f"{TRAIN} configs/side/plain_N5_T7.yaml name=side-f3-plain-3p2B-s0 train.total_bytes=3.2e9 train.eval_every=2000",
    f"{TRAIN} configs/side/latent_decay.yaml name=side-decay-s0 train.seed=0",
    f"{TRAIN} configs/side/latent_decay.yaml name=side-decay-s1 train.seed=1",
    f"[gpus=half] {TRAIN} configs/side/latent_N5_T7.yaml name=side-f10-latent-s0 train.total_bytes=1e10 train.eval_every=4000",
    f"[gpus=half] {TRAIN} configs/side/plain_N5_T7.yaml name=side-f10-plain-s0 train.total_bytes=1e10 train.eval_every=4000",
]


def preflight_lines(n_groups=4):
    cfgs = sorted(f"configs/{config_path(r)}" for r in RUNGS if RUNGS[r]["cfg"] not in ("p2_best", "p4_best"))
    cfgs = sorted(set(cfgs)) + sorted(f"configs/side/{p.name}" for p in (ROOT / "configs" / "side").glob("*.yaml"))
    groups = [cfgs[i::n_groups] for i in range(n_groups)]
    return ["# GPU preflight of every ladder config (~1-2 min per config). Run this first on a new pod.",
            "sh python scripts/preflight.py --stubs-only",
            *[f"python scripts/preflight.py --keep-stubs {' '.join(g)} name=preflight{i}" for i, g in enumerate(groups) if g],
            "barrier", "sh python scripts/preflight.py --remove-stubs"]


def write_jobs():
    d = ROOT / "scripts" / "ladder"
    d.mkdir(exist_ok=True)
    PHASES["preflight"] = preflight_lines()
    PHASES["side"] = SIDE
    PHASES["phase6"] = [
        "# Phase 6: latent chain of thought on synthetic arithmetic (after phase 3 picked P3).",
        "sh python -m latents.tasks --out data/task_arith",
        f"{TRAIN} configs/task/arith_p3.yaml name=L-T6a-s0 train.init_from=runs/$(python scripts/ladder.py name @P3 0)/final.pt",
        f"{TRAIN} configs/task/arith_r0.yaml name=L-T6b-s0 train.init_from=runs/L-R0-s0/final.pt",
        "barrier",
        "sh python scripts/latent_cot.py runs/L-T6a-s0 runs/L-T6b-s0 --k 0,2,4,8,16,oracle"]
    PHASES["evals"] = [
        "# Eval suite on milestone checkpoints (and the old 3B references if their final.pt is downloaded to runs/).",
        "sh python scripts/eval_suite.py fetch",
        *[f"python scripts/eval_suite.py run runs/{r}" for r in ("L-FLOOR3B-s0", "L-M1-s0", "L-M2-s0", "L-M3-s0")],
        "barrier",
        "sh python scripts/eval_suite.py table runs/L-FLOOR3B-s0 runs/L-M1-s0 runs/L-M2-s0 runs/L-M3-s0"]
    for name, lines in PHASES.items():
        (d / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print("wrote", (d / f"{name}.txt").relative_to(ROOT))


def main():
    cmd = sys.argv[1]
    if cmd == "check":
        ok, msg = check(sys.argv[2])
        print(msg)
        sys.exit(2 if ok is None else (0 if ok else 1))
    if cmd == "select":
        sys.exit(0 if select(sys.argv[2]) else 1)
    if cmd == "name":  # run name of a (possibly @-resolved) rung, for job files
        print(run_name(resolve(sys.argv[2]), int(sys.argv[3])))
        return
    if cmd == "report":
        report()
        return
    if cmd == "jobs":
        write_jobs()
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
