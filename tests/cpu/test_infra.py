"""Trainer, ladder checks, job runner, eval suite and latent-CoT scripts, end to end on CPU with a tiny model."""
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
import yaml

import tests.cpu_stubs as S
from latents import tasks, train as T
from latents.models import build_model
from latents import latent as latent_lib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
DATA = "data/_cpu_task"


@pytest.fixture(scope="module")
def task_data():
    tasks.write(ROOT / DATA, train_bytes=3e5, val_bytes=2e4, n_test=12, seed=0)
    yield ROOT / DATA
    shutil.rmtree(ROOT / DATA, ignore_errors=True)


def write_cfg(tmp, name, latent=None, **over):
    mc = S.tiny_mc(arch_layout=["m1", ["T2"], "m1"], dec_byte_emb=True)
    for k in ("vocab_size_padded", "vocab_size", "seq_len", "bytes_per_token"):
        mc.pop(k)
    mc.update(over.pop("model", {}))
    tc = dict(seq_len=96, batch_size=4, micro_batch=2, steps=6, total_bytes=1e4, lr=1e-3, wd=0.1, eval_every=3,
              eval_windows=4, final_eval_windows=4, log_every=1, seed=0, warmup_frac=0.2, decay_frac=0.2)
    tc.update(over.pop("train", {}))
    cfg = dict(name=name, group="cpu-test", data_dir=DATA, wandb=False, upload_hf=False, model=mc, train=tc)
    if latent:
        cfg["latent"] = latent
    p = Path(tmp) / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


LAT = dict(target="ema", ema_tau=0.99, ema_tau_end=0.99, head="mlp", decode_from_pred=True, beta_r=1.0, rollout_k=2,
           rollout_predictor_only=True)


def cleanup(*names):
    for n in names:
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)


def test_trainer_runs_and_checkpoint_reloads(tmp_path, task_data):
    name = "cpu-smoke"
    cleanup(name)
    T.main([str(write_cfg(tmp_path, name, LAT))])
    run = ROOT / "runs" / name
    sm = json.load(open(run / "summary.json"))
    assert sm["val/bpb"] > 0 and sm["train_flops"] > 0 and "val/lat/r2_l1" in sm and "val/edge_prec" in sm
    ck = torch.load(run / "final.pt", weights_only=False)
    m = build_model(ck["cfg"]["model"], device="cpu")
    m.load_state_dict(ck["model"], strict=True)
    latent_lib.build(ck["cfg"]["latent"], m, ck["cfg"]["model"]).load_state_dict(ck["latent"], strict=True)


def test_trainer_variants(tmp_path, task_data):
    variants = {
        "cpu-trunk": dict(latent=dict(LAT, detach_trunk=True, reg="var", reg_gamma="auto", reg_calib_frac=0.3),
                          model=dict(pool_k=4)),
        "cpu-surprise": dict(latent=dict(LAT, detach_trunk=True), model=dict(router="surprise", N=4, surprise=dict(warmup=0.3))),
        "cpu-nest": dict(latent=dict(LAT, detach_trunk=True),
                         model=dict(N=4, nest=dict(levels=3, enc="m1", pred="T2", router="learned", ratio=2,
                                                   topdown=True, tie="all", rollout_k=2, compute_penalty=0.03))),
        "cpu-floor": dict(latent=None, model=dict(floor=True)),
        "cpu-decay": dict(latent=dict(LAT, beta_schedule="cooldown")),
    }
    for name, v in variants.items():
        cleanup(name)
        T.main([str(write_cfg(tmp_path, name, v["latent"], model=v.get("model", {})))])
        sm = json.load(open(ROOT / "runs" / name / "summary.json"))
        assert sm["val/bpb"] == sm["val/bpb"] and sm["val/bpb"] > 0, (name, sm)
        cleanup(name)


def test_early_kill(tmp_path, task_data):
    cleanup("cpu-ref", "cpu-kill")
    (ROOT / "runs" / "cpu-ref").mkdir(parents=True)
    (ROOT / "runs" / "cpu-ref" / "metrics.jsonl").write_text(json.dumps({"step": 3, "val/bpb": 0.01}) + "\n")
    p = write_cfg(tmp_path, "cpu-kill", LAT, train=dict(kill_ref="cpu-ref", kill_margin=0.01, kill_after=0.0))
    with pytest.raises(SystemExit) as e:
        T.main([str(p)])
    assert e.value.code == 3
    assert "killed" in json.load(open(ROOT / "runs" / "cpu-kill" / "summary.json"))
    cleanup("cpu-ref", "cpu-kill")


def test_manual_kill_file(tmp_path, task_data, monkeypatch):
    """runs/<name>/KILL stops the run at its next eval like an early kill; a stale one from an earlier run doesn't."""
    name = "cpu-mkill"
    cleanup(name)
    run = ROOT / "runs" / name
    run.mkdir(parents=True)
    (run / "KILL").write_text("stale")
    orig = T.evaluate

    def evaluate(*a, **kw):  # the monitor drops the file while the run is going
        (run / "KILL").write_text("trunk info < 0 at 50%")
        return orig(*a, **kw)

    monkeypatch.setattr(T, "evaluate", evaluate)
    with pytest.raises(SystemExit) as e:
        T.main([str(write_cfg(tmp_path, name, LAT))])
    assert e.value.code == 3
    sm = json.load(open(run / "summary.json"))
    assert sm["killed"] == "manual: trunk info < 0 at 50%"
    cleanup(name)


def test_eval_suite_and_latent_cot(tmp_path, task_data, monkeypatch):
    import eval_suite as E
    import latent_cot as C
    name = "cpu-smoke"
    if not (ROOT / "runs" / name / "final.pt").exists():
        T.main([str(write_cfg(tmp_path, name, LAT))])
    sc = E.Scorer(ROOT / "runs" / name, device="cpu")
    res = sc.continuation([(b"Q: a=1; b=a+2; b=?\n", b"#### 3\n"), (b"", b"hello")])
    assert len(res) == 2 and res[0][1] == 7
    assert sc.text_bpb((ROOT / DATA / "val.u8").read_bytes(), max_windows=2) > 0
    # chunk task on the task val file standing in for FineWeb-Edu val
    fake = tmp_path / "root"
    (fake / "data" / "fwe").mkdir(parents=True)
    shutil.copy(ROOT / DATA / "val.u8", fake / "data" / "fwe" / "val.u8")
    monkeypatch.setattr(E, "ROOT", fake)
    monkeypatch.setattr(E, "EVAL", fake / "data" / "eval")
    E.task_chunks(sc, limit=8)
    out = E.task_perturb(sc, limit=2)
    assert set(E.PERTURB) <= set(out)
    r = C.evaluate(str(ROOT / "runs" / name), ["0", "2", "oracle"], n=3, n_cot=2)
    assert {"direct", "latent_k2", "latent_koracle", "byte_cot"} <= set(r["modes"])
    cleanup(name)


def test_ladder_checks_and_select(tmp_path, monkeypatch):
    import ladder as Lm
    monkeypatch.setattr(Lm, "ROOT", tmp_path)
    monkeypatch.setattr(Lm, "STATE", tmp_path / "runs" / "ladder_state.json")
    (tmp_path / "configs" / "ladder").mkdir(parents=True)

    def put(rid, seed, **kv):
        d = tmp_path / "runs" / f"L-{rid}-s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(kv))

    for s, v in ((0, 1.2740), (1, 1.2960)):
        put("R0", s, **{"val/bpb": v, "val/lat/ratio_tf_copy": 0.6})
    assert Lm.check("R0")[0] is True
    put("R1", 0, **{"val/bpb": 1.274, "val/lat/ratio_tf_copy": 0.6})
    assert Lm.check("R1")[0] is None  # seed 1 not finished
    put("R1", 1, **{"val/bpb": 1.330, "val/lat/ratio_tf_copy": 0.6})
    assert Lm.check("R1")[0] is True  # mean delta +0.017 <= TOL (0.03): within run-to-run noise
    put("R1", 1, **{"val/bpb": 1.380, "val/lat/ratio_tf_copy": 0.6})
    ok, msg = Lm.check("R1")
    assert ok is False and "FAIL" in msg  # mean delta +0.042 > TOL
    put("R1", 1, **{"val/bpb": 1.300, "val/lat/ratio_tf_copy": 0.6, "killed": "x"})
    assert Lm.check("R1")[0] is False
    for rid, b in (("R6", 1.40), ("R7", 1.39), ("R8", 1.37), ("R9", 1.45)):
        for s in (0, 1):
            put(rid, s, **{"val/bpb": b, "val/compression_L1/L0": 0.25, "val/edge_prec": 0.5})
    assert Lm.select("P2") == "R8"
    assert "r8_ste_router" in (tmp_path / "configs" / "ladder" / "p2_best.yaml").read_text()
    assert Lm.resolve("@P2") == "R8"
    # P3 falls back to the P2 winner when no rollout rung passed
    assert Lm.select("P3") == "R8"
    assert "p2_best.yaml" in (tmp_path / "configs" / "ladder" / "p3_best.yaml").read_text()
    # R7 killed: R8 / R9 are judged against R6 instead (R9 at +0.02 vs R6 passes, R8 at +0.12 does not)
    for s in (0, 1):
        put("R7", s, **{"val/bpb": 1.50, "val/compression_L1/L0": 0.25, "val/edge_prec": 0.5, "killed": "x"})
        put("R8", s, **{"val/bpb": 1.52, "val/compression_L1/L0": 0.25, "val/edge_prec": 0.4})
        put("R9", s, **{"val/bpb": 1.42, "val/compression_L1/L0": 0.25, "val/edge_prec": 0.3})
    assert Lm.check("R9")[0] is True and "R7 killed" in Lm.check("R9")[1]
    assert Lm.check("R8")[0] is False
    assert Lm.select("P2") == "R9"


def test_job_files_parse_and_dry_run(tmp_path, monkeypatch):
    import ladder as Lm
    import run_jobs as Q
    # generate into tmp and keep the dry-run queue logs there: the tests must be safe on a pod with a live queue
    monkeypatch.setattr(Lm, "ROOT", tmp_path)
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    (tmp_path / "configs").symlink_to(ROOT / "configs")
    (tmp_path / "scripts").mkdir()
    Lm.write_jobs()
    gen = sorted((tmp_path / "scripts" / "ladder").glob("*.txt"))
    assert {f.name for f in gen} == {f.name for f in (ROOT / "scripts" / "ladder").glob("*.txt")}
    for f in gen:
        assert f.read_text() == (ROOT / "scripts" / "ladder" / f.name).read_text(), f"{f.name}: run ladder.py jobs"
        items = Q.parse(f)
        assert items, f
        for it in items:
            if it["kind"] == "job" and "latents.train" in it["cmd"]:
                cfg = it["cmd"].split()[3]
                assert (ROOT / cfg).exists() or "best" in cfg, cfg
        assert Q.Queue(f, [0, 1], dry=True).run() == 0


def test_all_ladder_configs_build():
    """Every rung config loads and builds its model + latent objective (tiny widths, CPU)."""
    import preflight
    made = preflight.ensure_stubs()  # remove only these: a running preflight / ladder may own the others
    try:
        paths = sorted((ROOT / "configs" / "ladder").glob("*.yaml")) + sorted((ROOT / "configs" / "side").glob("*.yaml")) \
            + sorted((ROOT / "configs" / "task").glob("*.yaml"))
        for p in paths:
            cfg = T.load_cfg([str(p)])
            mc = dict(cfg["model"])
            small = S.tiny_mc()
            for k in ("d_model", "d_intermediate", "num_heads", "rotary_emb_dim", "window_size", "ssm_cfg"):
                mc[k] = small[k]
            mc.update(vocab_size_padded=256, vocab_size=256, seq_len=96, bytes_per_token=1.0)
            if cfg.get("latent"):
                mc["_latent_head"] = cfg["latent"].get("head", "linear")
            m = build_model(mc, device="cpu")
            if cfg.get("latent"):
                latent_lib.build(cfg["latent"], m, mc)
    finally:
        for p in made:
            p.unlink(missing_ok=True)


def test_data_parallel_two_processes(tmp_path, task_data):
    """torchrun with 2 CPU processes (gloo): gradients are averaged, eval is sharded, rank 0 writes the summary."""
    import os
    import subprocess
    name = "cpu-ddp"
    cleanup(name)
    p = write_cfg(tmp_path, name, dict(LAT, detach_trunk=True), model=dict(N=4, router="learned"),
                  train=dict(micro_batch=1))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--rdzv-backend=c10d",
                        "--rdzv-endpoint=localhost:29651", str(ROOT / "tests" / "cpu" / "ddp_runner.py"), str(p)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    sm = json.load(open(ROOT / "runs" / name / "summary.json"))
    assert sm["world"] == 2 and sm["val/bpb"] > 0
    cleanup(name)


def test_hold_waits_for_flag(tmp_path, monkeypatch):
    """`hold <name>` waits while logs/hold_<name> exists, then goes on; no file = no-op."""
    import threading
    import time
    import run_jobs as Q
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    monkeypatch.setattr(Q, "POLL", 0.05)
    monkeypatch.setattr(Q, "notify", lambda *a: None)
    (tmp_path / "logs").mkdir()
    flag = tmp_path / "logs" / "hold_T"
    flag.write_text("")
    f = tmp_path / "jobs.txt"
    f.write_text("hold T\nsh touch after_hold\nhold U\nsh touch after_u\n")
    threading.Timer(0.4, flag.unlink).start()
    t0 = time.time()
    assert Q.Queue(f, [0]).run() == 0
    assert time.time() - t0 >= 0.4
    assert (tmp_path / "after_hold").exists() and (tmp_path / "after_u").exists()
    ev = [json.loads(line)["event"] for line in open(tmp_path / "logs" / "queue_jobs.jsonl")]
    assert ev.count("hold") == 1 and ev.count("release") == 1


def test_gpu_guard_waits_for_room(tmp_path, monkeypatch):
    """With RUN_JOBS_MAX_PER_GPU / RUN_JOBS_MIN_FREE_GB set, a job waits until its GPU has room."""
    import run_jobs as Q
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    monkeypatch.setattr(Q, "POLL", 0.05)
    monkeypatch.setattr(Q, "notify", lambda *a: None)
    monkeypatch.setenv("RUN_JOBS_MAX_PER_GPU", "1")
    monkeypatch.setenv("RUN_JOBS_MIN_FREE_GB", "30")
    (tmp_path / "logs").mkdir()
    calls = []
    busy, room = {0: (3, 10.0), 1: (1, 12.0)}, {0: (1, 50.0), 1: (1, 12.0)}
    monkeypatch.setattr(Q, "gpu_load", lambda: calls.append(1) or (busy if len(calls) < 3 else room))
    f = tmp_path / "jobs.txt"
    f.write_text("true name=t1\n")
    assert Q.Queue(f, [0, 1]).run() == 0
    assert len(calls) >= 3
    ev = [json.loads(line) for line in open(tmp_path / "logs" / "queue_jobs.jsonl")]
    assert ev[0]["event"] == "start" and ev[0]["gpus"] == [0]
    # the launch left a reservation on GPU 0: another guarded queue now sees 2 processes there and must not stack
    q2 = Q.Queue(f, [0])
    assert q2.room(1) is False
    monkeypatch.setattr(Q, "RESERVE_S", 0)
    assert q2.room(1) is True
