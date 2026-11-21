"""Coconut on a byte H-Net (latents/coconut.py): example building, equivalence with the normal forward at stage 0,
gradients through the thoughts, no leak of the replaced steps, generation, and the train/eval scripts end to end."""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import tests.cpu_stubs as S
from latents import tasks
from latents.coconut import Coconut, START, STOP, example, n_written_steps
from latents.models import build_model, forward_lm

ROOT = Path(__file__).resolve().parents[2]


def make(feed="full", seed=0):
    torch.manual_seed(seed)
    mc = S.tiny_mc(seq_len=256, pool_k=0, N=4, arch_layout=["m1", ["T2"], "m1"])
    return build_model(mc, device="cpu"), mc


def probs(n=4, seed=0):
    rng = random.Random(seed)
    return [tasks.problem(rng, 2 + i % 3) for i in range(n)]


def test_examples_delete_the_replaced_steps():
    p = tasks.problem(random.Random(3), 3)
    assert n_written_steps(p["q"]) == 4
    b0, s0, f0, n0 = example(p, 0)
    assert s0 == -1 and n0 == 0 and b0.endswith(p["answer"].encode())
    b2, s2, f2, n2 = example(p, 2, c=2)
    assert b2[s2] == STOP and b2[s2 - 1] == START and n2 == 4 and f2 == s2 + 1
    first_two = p["cot"].split(" ")[:2]
    for step in first_two:
        assert step.encode() not in b2  # the replaced steps appear nowhere
    assert p["cot"].split(" ")[2].encode() in b2
    bf, sf, ff, nf = example(p, 9)
    assert bf[ff:] == p["answer"].encode() and nf == 4


def test_stage0_equals_the_normal_forward():
    m, mc = make()
    co = Coconut(m, mc)
    ps = probs()
    seqs = [example(p, 0)[0] for p in ps]
    with torch.no_grad():
        lg, _, _, _ = co.forward(seqs, [-1] * len(ps), [0] * len(ps))
        ref = forward_lm(m, co.batch(seqs)).logits
    assert torch.allclose(lg, ref, atol=1e-5)


@pytest.mark.parametrize("feed", ["full", "input"])
def test_loss_reaches_every_part_and_flows_through_the_thoughts(feed):
    m, mc = make()
    co = Coconut(m, mc, c=1, feed=feed)
    ps = probs()
    loss, st = co.loss(ps, [2] * len(ps))
    assert torch.isfinite(loss) and st["thoughts"] == 2
    for mod in (m.backbone.encoder, m.backbone.main_network, m.backbone.decoder, m.backbone.routing_module):
        g = torch.autograd.grad(loss, [p for p in mod.parameters() if p.requires_grad], retain_graph=True,
                                allow_unused=True)
        assert any(x is not None and x.abs().sum() > 0 for x in g)
    # the thoughts matter: changing them changes the loss on what follows
    exs = [example(p, 2) for p in ps]
    seqs = [e[0] for e in exs]
    with torch.no_grad():
        lg1, _, _, th = co.forward(seqs, [e[1] for e in exs], [e[3] for e in exs])
        lg2, _, _, _ = co.forward(seqs, [e[1] for e in exs], [e[3] for e in exs], thoughts=[t * 0 + 1.0 for t in th])
    assert (lg1 - lg2).abs().max() > 1e-4


def test_bytes_before_the_markers_do_not_see_the_thoughts():
    m, mc = make()
    co = Coconut(m, mc)
    ps = probs()
    exs = [example(p, 2) for p in ps]
    seqs = [e[0] for e in exs]
    with torch.no_grad():
        lg1, _, _, th = co.forward(seqs, [e[1] for e in exs], [e[3] for e in exs])
        lg2, _, _, _ = co.forward(seqs, [e[1] for e in exs], [e[3] for e in exs], thoughts=[t * 0 - 3.0 for t in th])
    for b, e in enumerate(exs):
        start = e[1] - 1
        assert (lg1[b, :start] - lg2[b, :start]).abs().max() < 1e-5  # question bytes: unaffected by thoughts


def test_generation_and_scoring_run():
    m, mc = make()
    co = Coconut(m, mc)
    ps = probs(3)
    outs = co.answer(ps, stage=None, max_new=12)
    assert len(outs) == 3 and all(n == n_written_steps(p["q"]) for (_, n), p in zip(outs, ps))
    outs0 = co.answer(ps, stage=0, max_new=12)
    assert all(n == 0 for _, n in outs0)


def test_train_and_eval_scripts(tmp_path):
    data = ROOT / "data" / "_cpu_coco"
    tasks.write(data, train_bytes=2e5, val_bytes=2e4, n_test=6, seed=0)
    base, name = "cpu-coco-base", "cpu-coco"
    for n in (base, name, *(f"{name}-stage{s}" for s in (1, 2))):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
    try:
        m, mc = make()
        (ROOT / "runs" / base).mkdir(parents=True, exist_ok=True)
        mc2 = dict(mc)
        torch.save({"model": m.state_dict(), "cfg": {"model": mc2, "train": {"seq_len": 256}}},
                   ROOT / "runs" / base / "final.pt")
        pre = ("import tests.cpu_stubs as S; S.install(); import runpy, sys; sys.argv=")
        r = subprocess.run([sys.executable, "-c", pre + repr(
            ["coconut_train.py", "--init", f"runs/{base}", f"--name={name}", "--stages", "2", "--stage_steps", "2",
             "--batch", "4", "--eval_n", "3", "--data_dir", "data/_cpu_coco", "--log_every", "1"])
            + "; runpy.run_path('scripts/coconut_train.py', run_name='__main__')"], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        sm = json.load(open(ROOT / "runs" / name / "summary.json"))
        assert set(sm["stages_eval"]) == {"stage1", "stage2"}
        assert (ROOT / "runs" / f"{name}-stage1" / "final.pt").exists()
        r = subprocess.run([sys.executable, "-c", pre + repr(
            ["coconut_eval.py", f"runs/{name}", "--n", "4", "--data_dir", "data/_cpu_coco"])
            + "; runpy.run_path('scripts/coconut_eval.py', run_name='__main__')"], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        ev = json.load(open(ROOT / "runs" / name / "coconut_eval.json"))
        assert {"all_thoughts", "no_thoughts", "written"} <= set(ev)
        assert ev["no_thoughts"]["mean_thoughts"] == 0
    finally:
        shutil.rmtree(data, ignore_errors=True)
        for n in (base, name, *(f"{name}-stage{s}" for s in (1, 2))):
            shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
