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
            ["coconut_train.py", "--init", f"runs/{base}", f"--name={name}", "--stages", "2", "--stage_steps", "2", "--wandb", "0", "--eval_every", "1", "--advance_at", "0.99",
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


def test_question_loss_and_carried_score():
    from latents.coconut import carried
    m, mc = make()
    co = Coconut(m, mc)
    ps = probs()
    l1, st = co.loss(ps, [2] * len(ps))  # default: the question's bytes are in the loss
    l0, _ = co.loss(ps, [2] * len(ps), q_loss=False)
    assert float(st["ce_question"]) > 0 and "cut_rate_question" in st
    assert abs(float(l0) - float(st["ce"]) - mc["alpha"] * float(st["ratio_loss"])) < 1e-5  # old loss = after the thoughts
    assert abs(float(l1) - float(l0)) > 1e-6
    # the markers are never targets: the question term stops before <start>
    p = ps[0]
    b, stop, first, n = example(p, 2)
    assert b[stop - 1] == START and first == stop + 1
    # carried value: the last replaced step's value opens the first written step, or is the answer
    p = tasks.problem(random.Random(3), 3)
    steps = p["cot"].strip().split(" ")
    v1 = int(steps[1].rstrip(".").split("=")[-1])
    good = (" ".join(steps[2:]) + "\n" + p["answer"], 2)
    assert carried([p], [good], 2) == {"exact": 1.0, "units": 1.0, "tens": 1.0}
    wrong = steps[2].replace(f"={v1}", f"={(v1 + 10) % 100}", 1)
    c = carried([p], [(wrong, 2)], 2)
    assert c["exact"] == 0.0 and c["units"] == 1.0
    assert carried([p], [(p["answer"], 4)], None)["exact"] == 1.0


def test_read_from_the_last_thought():
    """read="thought": the <stop> chunk is not a main-network position; its bytes are decoded from the output at the last
    thought. Stage 0 is unchanged, the loss reaches every part, and what is written after thinking does not depend on
    the <stop> chunk's own input."""
    m, mc = make()
    co = Coconut(m, mc, read="thought")
    ps = probs()
    seqs = [example(p, 0)[0] for p in ps]
    with torch.no_grad():
        lg, _, _, _ = co.forward(seqs, [-1] * len(ps), [0] * len(ps))
        assert torch.allclose(lg, forward_lm(m, co.batch(seqs)).logits, atol=1e-5)
    loss, st = co.loss(ps, [2] * len(ps))
    loss.backward()
    bb = m.backbone
    for mod in (bb.encoder, bb.main_network, bb.decoder, bb.routing_module):
        assert sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None) > 0
    exs = [example(p, 2) for p in ps]
    with torch.no_grad():
        l1, bp, _, th = co.forward([e[0] for e in exs], [e[1] for e in exs], [e[3] for e in exs])
        l2, _, _, _ = Coconut(m, mc, read="stop").forward([e[0] for e in exs], [e[1] for e in exs], [e[3] for e in exs])
        assert not torch.allclose(l1, l2, atol=1e-4)  # a different read-out from the default
        # one main-network position fewer than read="stop": the output at the last thought is the <stop> chunk's state
        th2 = [t + 0.5 * torch.randn_like(t) for t in th]
        l3, _, _, _ = co.forward([e[0] for e in exs], [e[1] for e in exs], [e[3] for e in exs], thoughts=th2)
        s0 = exs[0][1]
        assert not torch.allclose(l1[0, s0], l3[0, s0], atol=1e-5)  # the byte after <stop> is read from the thoughts
        assert torch.allclose(l1[0, : s0 - 1], l3[0, : s0 - 1], atol=1e-5)  # the question is untouched
    outs = co.answer(ps[:2], stage=None, max_new=12)
    assert len(outs) == 2 and all(isinstance(o[0], str) for o in outs)


def test_fresh_byte_state_after_start():
    """fresh=True: the byte-level encoder restarts at <start>, so its states from there on do not depend on the
    question; the whole forward still trains every part, and stage 0 is unchanged."""
    m, mc = make()
    co = Coconut(m, mc, read="thought", fresh=True)
    ps = probs()
    p, q = dict(ps[0]), dict(ps[0])
    c = p["q"][-4]  # the asked variable ("...; d=?\n"): another letter, close to <start> (the CPU stub's memory is short)
    q["q"] = p["q"][:-4] + ("x" if c != "x" else "y") + p["q"][-3:]
    exs = [example(p, 2), example(q, 2)]
    assert len(exs[0][0]) == len(exs[1][0]) and exs[0][0] != exs[1][0]
    x = co.batch([e[0] for e in exs])
    B, L = x.shape
    start = exs[0][1] - 1
    with torch.no_grad():
        cu_b, max_b = co.byte_seqs(B, L, [start, start], x.device)
        assert cu_b.tolist() == [0, start, L, L + start, 2 * L]
        h, _, _ = co.encode(x, torch.zeros(0, dtype=torch.long), cu_b, max_b)
        h = h.view(B, L, -1)
        assert torch.allclose(h[0, start:], h[1, start:], atol=1e-6)  # nothing of the question after <start>
        assert not torch.allclose(h[0, :start], h[1, :start], atol=1e-4)
        h0, _, _ = Coconut(m, mc).encode(x, torch.zeros(0, dtype=torch.long))
        h0 = h0.view(B, L, -1)
        assert float((h0[0, start] - h0[1, start]).abs().max()) > 1e-5  # default: the question reaches <start>
        assert float((h[0, start] - h[1, start]).abs().max()) < 1e-7
        seqs = [example(pp, 0)[0] for pp in ps]
        lg, _, _, _ = co.forward(seqs, [-1] * len(ps), [0] * len(ps))
        assert torch.allclose(lg, forward_lm(m, co.batch(seqs)).logits, atol=1e-5)
    loss, st = co.loss(ps, [2] * len(ps))
    loss.backward()
    bb = m.backbone
    for mod in (bb.encoder, bb.main_network, bb.decoder, bb.routing_module):
        assert sum(float(pp.grad.abs().sum()) for pp in mod.parameters() if pp.grad is not None) > 0
    assert len(co.answer(ps[:2], stage=None, max_new=8)) == 2


def test_ratio_loss_per_part():
    m, mc = make()
    co = Coconut(m, mc)
    ps = probs()
    torch.manual_seed(0)
    l_row, st_row = co.loss(ps, [2] * len(ps))
    l_parts, st_parts = co.loss(ps, [2] * len(ps), ratio="parts")
    assert abs(float(st_row["ce"]) - float(st_parts["ce"])) < 1e-6 and float(st_row["ratio_loss"]) != float(st_parts["ratio_loss"])
    l_parts.backward()
    assert sum(float(p.grad.abs().sum()) for p in m.backbone.routing_module.parameters() if p.grad is not None) > 0


def test_grounded_thoughts_take_the_reasoning_chunks_as_count_and_target():
    """ground > 0: one thought per chunk of the deleted reasoning (the model's own chunking of the written example), each
    trained toward that chunk's latent; the target is a constant, the loss reaches the main network through the thoughts,
    and generation uses the same count."""
    m, mc = make()
    co = Coconut(m, mc, read="thought", feed="input", ground=1.0)
    ps = probs()
    tg = co.reasoning_chunks(ps, [2] * len(ps))
    assert all(t.ndim == 2 and t.shape[1] == mc["d_model"][0] and not t.requires_grad for t in tg)
    full = co.reasoning_chunks(ps, [99] * len(ps))
    assert all(f.shape[0] >= t.shape[0] for f, t in zip(full, tg)) and sum(f.shape[0] for f in full) > sum(t.shape[0] for t in tg)
    assert all(t.shape[0] == 0 for t in co.reasoning_chunks(ps, [0] * len(ps)))
    loss, st = co.loss(ps, [2] * len(ps))
    assert abs(float(st["thoughts"]) - sum(t.shape[0] for t in tg) / len(ps)) < 1e-6 and float(st["ground_loss"]) > 0
    l0, _ = Coconut(m, mc, read="thought", feed="input").loss(ps, [2] * len(ps))
    m.zero_grad()
    (loss - 0).backward()
    bb = m.backbone
    assert sum(float(p.grad.abs().sum()) for p in bb.main_network.parameters() if p.grad is not None) > 0
    # the grounding term alone still trains the main network and the encoder inputs it reads, not the decoder
    m.zero_grad()
    exs = [example(p, 2) for p in ps]
    _, _, _, th = co.forward([e[0] for e in exs], [e[1] for e in exs], [t.shape[0] for t in tg])
    pr = torch.cat([t[:, : mc["d_model"][0]] for t in th])
    F.smooth_l1_loss(pr, torch.cat(tg)).backward()
    assert sum(float(p.grad.abs().sum()) for p in bb.main_network.parameters() if p.grad is not None) > 0
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in bb.decoder.parameters())
    outs = co.answer(ps[:2], stage=None, max_new=8)
    assert [o[1] for o in outs] == [t.shape[0] for t in full[:2]]
