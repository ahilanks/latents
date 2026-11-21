"""Hidden-state feedback (model.feedback, latents/models.py main_pass): the main network's last-layer output at one
chunk is added to its input at the next chunk. Starts as the plain model, stays causal, keeps rows apart, trains."""
import json
import shutil
from pathlib import Path

import torch

import tests.cpu_stubs as S
from latents import tasks, train as T
from latents.coconut import Coconut, example
from latents.models import build_model, forward_lm, param_groups

ROOT = Path(__file__).resolve().parents[2]
DATA = "data/_cpu_task_fb"


def make(seed=0, live=True, **fb):
    torch.manual_seed(seed)
    mc = S.tiny_mc(seq_len=96, feedback=dict(dict(passes=2), **fb))
    m = build_model(mc, device="cpu")
    if live:
        last = m.feedback.net if isinstance(m.feedback.net, torch.nn.Linear) else m.feedback.net[-1]
        torch.nn.init.normal_(last.weight, std=0.3)
    return m, mc, torch.randint(0, 256, (3, 96))


def test_starts_as_the_plain_model():
    m, mc, x = make(live=False)
    with torch.no_grad():
        a = forward_lm(m, x).logits
        fb = m.feedback
        del m.feedback
        b = forward_lm(m, x).logits
        m.feedback = fb
    assert torch.allclose(a, b, atol=1e-6)


def test_feedback_changes_the_output_and_stays_causal_and_rowwise():
    m, mc, x = make()
    with torch.no_grad():
        a = forward_lm(m, x).logits
        fb = m.feedback
        del m.feedback
        plain = forward_lm(m, x).logits
        m.feedback = fb
        assert not torch.allclose(a, plain, atol=1e-4)
        y = x.clone()
        y[0, 60:] = (y[0, 60:] + 7) % 256  # later bytes of row 0
        b = forward_lm(m, y).logits
    assert torch.allclose(a[0, :60], b[0, :60], atol=1e-5)  # nothing from the future
    assert not torch.allclose(a[0, 60:], b[0, 60:], atol=1e-4)
    assert torch.allclose(a[1:], b[1:], atol=1e-5)  # other rows untouched


def test_gradient_reaches_the_feedback_and_every_part():
    m, mc, x = make(hidden=24)
    out = forward_lm(m, x)
    torch.nn.functional.cross_entropy(out.logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()).backward()
    bb = m.backbone
    for mod in (m.feedback, bb.encoder, bb.main_network, bb.decoder, bb.routing_module):
        assert sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None) > 0
    groups, mults = param_groups(m, mc, 0.1)
    g = next(g for g in groups if any(n.startswith("feedback.") for n in g["names"]) and g["weight_decay"] > 0)
    assert g["lr_mult"] == mults[-1]


def test_three_passes_and_coconut_written_mode_use_the_same_pass():
    m, mc, x = make(passes=3)
    co = Coconut(m, mc)
    ps = [tasks.problem(__import__("random").Random(i), 2 + i % 3) for i in range(4)]
    seqs = [example(p, 0)[0] for p in ps]
    with torch.no_grad():
        lg, _, _, _ = co.forward(seqs, [-1] * len(ps), [0] * len(ps))
        ref = forward_lm(m, co.batch(seqs)).logits
        m._feedback_cfg["passes"] = 2
        two = forward_lm(m, co.batch(seqs)).logits
    assert torch.allclose(lg, ref, atol=1e-5) and not torch.allclose(ref, two, atol=1e-4)


def test_trainer_runs_with_feedback_and_reloads(tmp_path):
    import yaml
    tasks.write(ROOT / DATA, train_bytes=3e5, val_bytes=2e4, n_test=12, seed=0)
    name = "cpu-feedback"
    shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
    mc = S.tiny_mc(feedback=dict(passes=2))
    for k in ("vocab_size_padded", "vocab_size", "seq_len", "bytes_per_token"):
        mc.pop(k)
    tc = dict(seq_len=96, batch_size=4, micro_batch=2, steps=6, total_bytes=1e4, lr=1e-3, wd=0.1, eval_every=3,
              eval_windows=4, final_eval_windows=4, log_every=1, seed=0, warmup_frac=0.2, decay_frac=0.2)
    p = Path(tmp_path) / "c.yaml"
    p.write_text(yaml.safe_dump(dict(name=name, group="cpu-test", data_dir=DATA, wandb=False, upload_hf=False,
                                     model=mc, train=tc)))
    try:
        T.main([str(p)])
        sm = json.load(open(ROOT / "runs" / name / "summary.json"))
        assert sm["val/bpb"] > 0
        ck = torch.load(ROOT / "runs" / name / "final.pt", weights_only=False)
        m2 = build_model(ck["cfg"]["model"], device="cpu")
        m2.load_state_dict(ck["model"], strict=True)
        assert float(m2.feedback.net.weight.abs().sum()) > 0  # it trained away from zero
    finally:
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
        shutil.rmtree(ROOT / DATA, ignore_errors=True)


def test_loop_is_exact_and_equals_enough_passes():
    """mode=loop: the last-layer output at chunk t goes into the same network's input at chunk t+1. It must equal the
    k-pass version once k reaches the longest row (the passes are exact for a row's first k chunks)."""
    m, mc, x = make()
    with torch.no_grad():
        m._feedback_cfg = {"mode": "loop"}
        out = forward_lm(m, x)
        loop = out.logits
        T = int((out.cu1[1:] - out.cu1[:-1]).max())
        m._feedback_cfg = {"passes": T}
        full = forward_lm(m, x).logits
        m._feedback_cfg = {"passes": 2}
        two = forward_lm(m, x).logits
        y = x.clone()
        y[0, 60:] = (y[0, 60:] + 7) % 256
        m._feedback_cfg = {"mode": "loop"}
        b = forward_lm(m, y).logits
    assert T > 3 and torch.allclose(loop, full, atol=2e-4)
    assert not torch.allclose(loop, two, atol=1e-3)  # two passes are only an approximation
    assert torch.allclose(loop[0, :60], b[0, :60], atol=1e-5) and torch.allclose(loop[1:], b[1:], atol=1e-5)


def test_loop_trains_and_coconut_written_mode_matches_on_the_real_bytes():
    import random
    m, mc, _ = make()
    m._feedback_cfg = {"mode": "loop"}
    co = Coconut(m, mc)
    ps = [tasks.problem(random.Random(i), 2 + i % 3) for i in range(4)]
    seqs = [example(p, 0)[0] for p in ps]
    with torch.no_grad():
        lg, _, _, _ = co.forward(seqs, [-1] * len(ps), [0] * len(ps))  # padding chunks are skipped by the loop
        ref = forward_lm(m, co.batch(seqs)).logits
    for b, sq in enumerate(seqs):
        assert torch.allclose(lg[b, : len(sq)], ref[b, : len(sq)], atol=1e-4)
    loss, _ = co.loss(ps, [0] * len(ps))
    loss.backward()
    for mod in (m.feedback, m.backbone.encoder, m.backbone.main_network, m.backbone.decoder):
        assert sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None) > 0


def test_train_script_adds_the_loop_to_a_plain_model(tmp_path):
    import subprocess
    import sys
    base, name = "cpu-loop-base", "cpu-loop"
    tasks.write(ROOT / "data/_cpu_loop", train_bytes=2e4, val_bytes=2e3, n_test=6, seed=0)
    try:
        torch.manual_seed(0)
        mc = S.tiny_mc(seq_len=256, pool_k=0, N=4)
        m = build_model(mc, device="cpu")
        (ROOT / "runs" / base).mkdir(parents=True, exist_ok=True)
        torch.save({"model": m.state_dict(), "cfg": {"model": dict(mc), "train": {"seq_len": 256}}}, ROOT / "runs" / base / "final.pt")
        pre = "import tests.cpu_stubs as S; S.install(); import runpy, sys; sys.argv="
        r = subprocess.run([sys.executable, "-c", pre + repr(
            ["coconut_train.py", "--init", f"runs/{base}", f"--name={name}", "--feedback", "loop", "--first_stage", "0",
             "--stages", "0", "--stage_steps", "3", "--batch", "4", "--eval_n", "3", "--data_dir", "data/_cpu_loop",
             "--log_every", "1", "--wandb", "0"])
            + "; runpy.run_path('scripts/coconut_train.py', run_name='__main__')"], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        ck = torch.load(ROOT / "runs" / name / "final.pt", weights_only=False)
        assert ck["cfg"]["model"]["feedback"] == {"mode": "loop"}
        m2 = build_model(ck["cfg"]["model"], device="cpu")
        m2.load_state_dict(ck["model"], strict=True)
        assert float(m2.feedback.net.weight.detach().abs().sum()) > 0
    finally:
        for n in (base, name, f"{name}-stage0"):
            shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
        shutil.rmtree(ROOT / "data/_cpu_loop", ignore_errors=True)
