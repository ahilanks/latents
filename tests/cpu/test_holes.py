"""The hole recipe (latents/holes.py, model.holes, model.feedback with plain updates, health numbers, reasoning in the
holes, the language task): CPU checks with the tiny H-Net."""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import tests.cpu_stubs as S
import latents.models as M
from latents import health as HL
from latents import holes as H
from latents import tasks, tasks_logic
from latents import train as T
from latents.models import build_model, feedback_mode, forward_lm, model_forward, ratio_loss

ROOT = Path(__file__).resolve().parents[2]
WORDS = ("the of and to in is was for on that with as by at from his her an be this which or are have had "
         "not but one all were their there been when who will more no if out so said what up its about than into").split()


def web_like(n, seed=0):
    rng = random.Random(seed)
    out, k = [], 0
    while k < n:
        doc = bytes([254]) + " ".join(rng.choice(WORDS) for _ in range(rng.randint(20, 120))).encode() + b".\n"
        out.append(doc)
        k += len(doc)
    return np.frombuffer(b"".join(out), dtype=np.uint8)


def make(seed=0, live=True, **fb):
    torch.manual_seed(seed)
    mc = S.tiny_mc(seq_len=128, feedback=dict(dict(passes=4), **fb), holes={"mark": 3})
    m = build_model(mc, device="cpu")
    if live:
        torch.nn.init.normal_(m.feedback.net.weight, std=0.3)
    return m, mc


# ----------------------------------------------------------------------------- data
def test_punch_deletes_about_a_tenth_of_words_in_runs_of_one_to_three():
    w = web_like(400_000)
    rng = np.random.default_rng(0)
    tot_w = tot_d = 0
    runs = []
    for i in range(0, 300_000, 3000):
        src = w[i:i + 3000]
        out, n, d = H.punch(src, rng, 0.1, 3)
        tot_w, tot_d = tot_w + n, tot_d + d
        h = out == 3
        assert int(h.sum()) == d  # one marker per deleted word
        edges = np.flatnonzero(np.diff(np.concatenate(([0], h.astype(np.int8), [0]))))
        runs += list(edges[1::2] - edges[::2])
        # the text with the marker runs put back equals the source; separators and the first / last word are kept
        st, en = H.word_spans(src)
        assert out[: en[0]].tobytes() == src[: en[0]].tobytes()
        assert out[len(out) - (len(src) - st[-1]):].tobytes() == src[st[-1]:].tobytes()
        assert (out == 254).sum() == (src == 254).sum()
    assert 0.08 < tot_d / tot_w < 0.12, tot_d / tot_w
    assert set(runs) <= {1, 2, 3} and runs.count(1) > runs.count(2) > runs.count(3) > 0


def test_punch_keeps_the_text_around_a_hole():
    src = np.frombuffer(b"\xfeone two three four five six seven eight nine ten", dtype=np.uint8)
    for seed in range(50):
        out, n, d = H.punch(src, np.random.default_rng(seed), 0.4, 3)
        words = [s for s in out.tobytes()[1:].split(b" ")]
        kept = [w_ for w_ in words if 3 not in w_]
        assert all(w_ in src.tobytes() for w_ in kept)
        assert all(set(w_) == {3} for w_ in words if 3 in w_)  # a run of markers is its own word
        assert out[1:4].tobytes() == b"one" and out[-3:].tobytes() == b"ten"


def test_holed_window_has_the_requested_length():
    w = web_like(100_000)
    rng = np.random.default_rng(1)
    for i in (0, 5000, 40_000):
        x = H.holed_window(w, i, 777, rng)
        assert x.shape == (777,) and (x == 3).sum() > 0


def test_after_hole_targets_cover_the_next_word():
    x = torch.tensor([list(b"ab \x03\x03 cd ef \x03gh i")])
    m = H.after_hole_targets(x)[0].tolist()
    # targets at t predict byte t+1: after the run ending at 4 -> bytes 5..7 (" cd"); after 12 -> bytes 13..14 ("gh")
    assert [t for t, v in enumerate(m) if v] == [4, 5, 6, 12, 13]


# ----------------------------------------------------------------------------- model
def test_a_hole_is_one_chunk_and_its_input_is_only_what_is_fed_back(monkeypatch):
    m, mc = make()
    x = torch.tensor([list(b"\xfehello world \x03\x03 and more text here \x03 end of it all") + [32] * 4])
    seen = {}
    real = M.main_pass

    def spy(mm, hp, cu1, max1, valid=None):
        seen["hp"] = hp.detach().clone()
        return real(mm, hp, cu1, max1, valid)
    monkeypatch.setattr(M, "main_pass", spy)
    with torch.no_grad():
        out = forward_lm(m, x, opts={"holes": 3})
    bm = out.bpred[0].boundary_mask
    hole = (x.flatten() == 3)
    after = torch.nn.functional.pad(hole[:-1], (1, 0))
    assert bool(bm[hole].all()) and bool(bm[after].all())
    assert torch.equal(out.extras["forced"], hole | after)
    hc_hole = hole[bm]
    D0 = mc["d_model"][0]
    assert int(hc_hole.sum()) == 3 and float(seen["hp"][hc_hole, :D0].abs().sum()) == 0.0
    assert float(seen["hp"][~hc_hole, :D0].abs().sum()) > 0


def test_without_markers_the_forward_is_unchanged():
    m, mc = make()
    x = torch.randint(4, 254, (2, 96))
    with torch.no_grad():
        a = forward_lm(m, x).logits
        b = forward_lm(m, x, opts={"holes": 3}).logits
    assert torch.equal(a, b)


def test_ratio_loss_leaves_out_forced_cuts():
    m, mc = make()
    x = torch.tensor([list(b"\xfesome words \x03\x03 more words \x03 the end, and then the rest....")])
    out = forward_lm(m, x, opts={"holes": 3})
    bp, fr = out.bpred[0], out.extras["forced"]
    N = 4.0
    Fm, G = bp.boundary_mask[~fr].float().mean(), bp.boundary_prob[..., -1][~fr].float().mean()
    want = N / (N - 1) * ((N - 1) * Fm * G + (1 - Fm) * (1 - G))
    assert torch.allclose(ratio_loss(bp, N, keep=~fr), want)
    assert not torch.allclose(ratio_loss(bp, N), want)


def test_feedback_modes_one_round_is_the_plain_model_and_the_loop_is_exact_with_holes():
    m, mc = make()
    r0 = list(b"\xfeab cd \x03 ef gh ij \x03\x03\x03 kl mn op qr st") * 2
    r1 = list(b"\xfe uv \x03 wx yz") * 5
    x = torch.tensor([r0, r1 + [32] * (len(r0) - len(r1))])
    with torch.no_grad():
        with feedback_mode(m, rounds=1):
            one = forward_lm(m, x, opts={"holes": 3}).logits
        fb = m.feedback
        del m.feedback
        plain = forward_lm(m, x, opts={"holes": 3}).logits
        m.feedback = fb
        with feedback_mode(m, mode="loop"):
            out = forward_lm(m, x, opts={"holes": 3})
        Tn = int((out.cu1[1:] - out.cu1[:-1]).max())
        with feedback_mode(m, rounds=Tn):
            full = forward_lm(m, x, opts={"holes": 3}).logits
        four = forward_lm(m, x, opts={"holes": 3}).logits
    assert m._feedback_cfg == {"passes": 4}  # restored
    assert torch.allclose(one, plain, atol=1e-6)
    assert torch.allclose(out.logits, full, atol=2e-4) and not torch.allclose(out.logits, four, atol=1e-4)


def test_gradient_reaches_every_part_through_the_rounds_and_the_holes():
    m, mc = make(hidden=0)
    x = torch.tensor([list(b"\xfeab cd \x03 ef gh ij \x03\x03 kl mn op qr st uv wx")] * 2)
    out = model_forward(m, x, mc)
    keep = H.target_mask(x[:, 1:], 3)
    ce = torch.nn.functional.cross_entropy(out.logits[:, :-1].flatten(0, 1), x[:, 1:].flatten(), reduction="none")
    (ce * keep.flatten()).sum().backward()
    bb = m.backbone
    for mod in (m.feedback, bb.encoder, bb.main_network, bb.decoder, bb.routing_module):
        assert sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None) > 0


def test_batch_health_numbers():
    m, mc = make()
    x = torch.randint(4, 254, (3, 128))
    x[:, 40] = 3
    res = HL.batch_health(m, x[:, :-1], x[:, 1:], mc)
    for k in ("main_variation", "main_rank", "swap_nats", "enc_cos_next", "fb_weight", "fb_share", "round_diff",
              "round_bpb_diff", "loop_off_bpb"):
        assert k in res and np.isfinite(res[k]), k
    assert res["round_diff"] > 0 and res["fb_weight"] > 0


def test_question_mask():
    x = torch.tensor([list(b"\xfeQ: a=1; b=a+2; b=?\na=1. b=3.\n#### 3\n\xfeQ: x\nrest")])
    q = HL.question_mask(x)[0].tolist()
    s = bytes(x[0].tolist())
    assert bytes(b for b, f in zip(s, q) if f) == b"Q: a=1; b=a+2; b=?\nQ: x\n"


# ----------------------------------------------------------------------------- reasoning data
def test_steps_and_counts_agree_for_both_tasks():
    rng = random.Random(0)
    for _ in range(50):
        p = tasks.sample(rng)
        assert H.count_steps(p["q"]) == len(H.steps_of(p["cot"])) == p["steps"] + 1
        p = tasks_logic.sample(rng)
        assert H.count_steps(p["q"]) == len(H.steps_of(p["cot"])) == p["steps"] + 1
        assert p["answer"] == f"#### {p['value']}\n"


def test_language_task_is_answerable_from_the_rules():
    rng = random.Random(1)
    yes = 0
    for _ in range(300):
        p = tasks_logic.problem(rng, rng.randint(2, 6))
        rules = dict(r.split(" is a ") for r in p["q"][3:].split(". ") if r.startswith("Every "))
        rules = {k[6:]: v.rstrip(".") for k, v in rules.items()}
        name, start = p["q"].split(". Is ")[0].split(". ")[-1].split(" is a ")
        target = p["q"].split(". Is ")[1].split(" a ")[1].rstrip("?\n")
        c, chain = start, [start]
        while c in rules:
            c = rules[c]
            chain.append(c)
        assert ("yes" if target in chain else "no") == p["value"]
        assert p["cot"] == " ".join(f"{name} is a {c}." for c in chain) + "\n"
        yes += p["value"] == "yes"
    assert 120 < yes < 180


def test_example_hides_steps_with_three_markers_each():
    p = tasks.problem(random.Random(3), 3)
    b0, n0 = H.example(p, 0)
    assert b0 == bytes([254]) + (p["q"] + p["cot"] + p["answer"]).encode() and n0 == len(p["q"]) + 1
    b2, n2 = H.example(p, 2)
    rest = p["cot"][H.steps_of(p["cot"])[2][0]:]
    assert b2 == bytes([254]) + p["q"].encode() + b"\x03" * 6 + (rest + p["answer"]).encode()
    ball, nall = H.example(p, 4)
    assert ball.endswith(b"\x03" * 12 + p["answer"].encode())
    bn, _ = H.example(p, 4, marks=False)
    assert bn == bytes([254]) + (p["q"] + p["answer"]).encode()


def test_generation_exact_loop_equals_enough_rounds():
    m, mc = make()
    probs = [tasks.problem(random.Random(i), 2) for i in range(3)]
    pr = H.prompts_for(probs, None)
    a = H.generate(m, mc, pr, max_new=6, exact=True)
    b = H.generate(m, mc, pr, max_new=6, exact=False, rounds=64)
    assert a == b and all(len(t.encode("utf-8", "surrogateescape")) > 0 for t in a)


# ----------------------------------------------------------------------------- trainers
def test_pretraining_with_holes_and_the_loop_runs_logs_health_and_reloads(tmp_path):
    name, data = "cpu-holes", ROOT / "data/_cpu_holes"
    data.mkdir(parents=True, exist_ok=True)
    web_like(400_000, 1).tofile(data / "train.u8")
    web_like(40_000, 2).tofile(data / "val.u8")
    shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
    mc = S.tiny_mc(feedback=dict(passes=4, plain_frac=0.5), holes=dict(mark=3))
    for k in ("vocab_size_padded", "vocab_size", "seq_len", "bytes_per_token"):
        mc.pop(k)
    tc = dict(seq_len=128, batch_size=4, micro_batch=2, steps=8, total_bytes=1e4, lr=1e-3, wd=0.1, eval_every=4,
              eval_windows=4, final_eval_windows=4, log_every=1, seed=0, warmup_frac=0.2, decay_frac=0.2,
              holes=dict(rate=0.1, max_run=3), hole_eval_windows=4)
    p = Path(tmp_path) / "c.yaml"
    p.write_text(yaml.safe_dump(dict(name=name, group="cpu-test", data_dir="data/_cpu_holes", wandb=False,
                                     upload_hf=False, model=mc, train=tc)))
    try:
        T.main([str(p)])
        rows = [json.loads(l) for l in open(ROOT / "runs" / name / "metrics.jsonl")]
        tr = [r for r in rows if "train/ce" in r]
        assert 0 < tr[-1]["train/plain_updates"] < 1 and all(r["train/hole_share"] > 0 for r in tr)
        ev = [r for r in rows if "val/bpb" in r][-1]
        for k in ("val/main_variation", "val/main_rank", "val/swap_nats", "val/enc_cos_next", "val/fb_weight",
                  "val/round_diff", "val/loop_off_bpb", "val_holes/bpb_after", "val_holes/bpb_after_off",
                  "val_holes/cut_rate_near", "val/router_rank_q", "val/cuts_after_space"):
            assert k in ev and np.isfinite(ev[k]), k
        sm = json.load(open(ROOT / "runs" / name / "summary.json"))
        assert sm["val/bpb"] > 0 and "val_holes/loop_gain_after" in sm
        assert "□" in open(ROOT / "runs" / name / "chunks_holes.txt").read()
        ck = torch.load(ROOT / "runs" / name / "final.pt", weights_only=False)
        m2 = build_model(ck["cfg"]["model"], device="cpu")
        m2.load_state_dict(ck["model"], strict=True)
        assert float(m2.feedback.net.weight.abs().sum()) > 0
    finally:
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
        shutil.rmtree(data, ignore_errors=True)


def _script(args, script):
    pre = "import tests.cpu_stubs as S; S.install(); import runpy, sys; sys.argv="
    return subprocess.run([sys.executable, "-c", pre + repr([script] + args) + f"; runpy.run_path('{script}', run_name='__main__')"],
                          cwd=ROOT, capture_output=True, text=True)


def test_reasoning_in_the_holes_trainer_and_eval_run():
    base, names = "cpu-holes-base", ["cpu-hr", "cpu-hr-all", "cpu-hr-logic"]
    tasks.write(ROOT / "data/_cpu_hr", train_bytes=2e4, val_bytes=2e3, n_test=6, seed=0)
    tasks_logic.write(ROOT / "data/_cpu_hrl", train_bytes=2e4, val_bytes=2e3, n_test=4, seed=0)
    try:
        m, mc = make()
        (ROOT / "runs" / base).mkdir(parents=True, exist_ok=True)
        torch.save({"model": m.state_dict(), "cfg": {"model": dict(mc), "train": {"seq_len": 128}}},
                   ROOT / "runs" / base / "final.pt")
        common = ["--init", f"runs/{base}", "--stage_steps", "3", "--batch", "4", "--eval_n", "3", "--eval_every", "2",
                  "--log_every", "1", "--wandb", "0", "--web_windows", "0"]
        for nm, extra in (("cpu-hr", ["--stages", "1,2", "--data_dir", "data/_cpu_hr"]),
                          ("cpu-hr-all", ["--stages", "all", "--marks", "0", "--data_dir", "data/_cpu_hr"]),
                          ("cpu-hr-logic", ["--task", "logic", "--stages", "2", "--data_dir", "data/_cpu_hrl"])):
            r = _script(["--name", nm] + common + extra, "scripts/holes_train.py")
            assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        sm = json.load(open(ROOT / "runs/cpu-hr/summary.json"))
        assert set(sm["stages_eval"]) == {"stage1", "stage2"} and "all_hidden" in sm["stages_eval"]["stage2"]
        rows = [json.loads(l) for l in open(ROOT / "runs/cpu-hr/metrics.jsonl")]
        hl = [r["health"] for r in rows if "health" in r]
        assert hl and all(0 <= h["cuts_kept_from_start"] <= 1 and "cut_rate_question" in h for h in hl)
        ck = torch.load(ROOT / "runs/cpu-hr-stage2/final.pt", weights_only=False)
        assert ck["cfg"]["holes_train"]["stage"] == 2
        r = _script(["runs/cpu-hr", "--modes", "written,hidden,stage1,nothoughts", "--n", "3", "--data_dir", "data/_cpu_hr"],
                    "scripts/holes_eval.py")
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        ev = json.load(open(ROOT / "runs/cpu-hr/holes_eval.json"))
        assert all(k in ev for k in ("written", "hidden", "stage1", "nothoughts")) and ev["loop"] == "exact"
    finally:
        for n in [base] + names + ["cpu-hr-stage1", "cpu-hr-stage2", "cpu-hr-all-all", "cpu-hr-logic-stage2"]:
            shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
        shutil.rmtree(ROOT / "data/_cpu_hr", ignore_errors=True)
        shutil.rmtree(ROOT / "data/_cpu_hrl", ignore_errors=True)


# ----------------------------------------------------------------------------- loop fixes (2026-11-24)
def test_centred_loop_is_exact_and_removes_any_constant():
    m, mc = make(center=True)
    m._feedback_cfg = dict(m._feedback_cfg, center=True)
    x = torch.tensor([list(b"\xfeab cd \x03 ef gh ij \x03\x03\x03 kl mn op qr st") * 2] * 2)
    with torch.no_grad():
        with feedback_mode(m, mode="loop"):
            out = forward_lm(m, x, opts={"holes": 3})
        Tn = int((out.cu1[1:] - out.cu1[:-1]).max())
        with feedback_mode(m, rounds=Tn):
            full = forward_lm(m, x, opts={"holes": 3}).logits
        assert torch.allclose(out.logits, full, atol=2e-4)
        # a feedback layer that only outputs a constant (weights 0, bias b) adds nothing once centred (text without
        # holes: at an untrained hole slot the input is exactly zero and the first norm blows up float rounding)
        torch.nn.init.zeros_(m.feedback.net.weight)
        torch.nn.init.normal_(m.feedback.net.bias, std=1.0)
        y = torch.randint(4, 254, (2, 96))
        a = forward_lm(m, y).logits
        with feedback_mode(m, rounds=1):
            b = forward_lm(m, y).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_causal_row_mean():
    g = torch.tensor([[1.0], [3.0], [5.0], [10.0], [20.0]])
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    assert torch.allclose(M.causal_row_mean(g, cu).flatten(), torch.tensor([1.0, 2.0, 3.0, 10.0, 15.0]))


def test_slow_loop_learning_rate():
    from latents.models import param_groups
    m, mc = make(lr_mult=0.1)
    groups, mults = param_groups(m, mc, 0.1)
    g = next(g for g in groups if any(n.startswith("feedback.") for n in g["names"]))
    assert abs(g["lr_mult"] - 0.1 * mults[-1]) < 1e-9


def test_paper_form_of_the_loop_rms_no_bias():
    """norm=rms, bias=False (Latent Feedback eq. 1): starts as the plain model, has no constant term, the loop is
    exact, and every part gets gradient."""
    m, mc = make(live=False, norm="rms", bias=False)
    assert isinstance(m.feedback.norm, torch.nn.RMSNorm) and m.feedback.net.bias is None
    x = torch.tensor([list(b"\\xfeab cd \\x03 ef gh ij \\x03\\x03\\x03 kl mn op qr st") * 2] * 2)
    with torch.no_grad():
        a = forward_lm(m, x, opts={"holes": 3}).logits
        with feedback_mode(m, rounds=1):
            b = forward_lm(m, x, opts={"holes": 3}).logits
    assert torch.equal(a, b)
    torch.nn.init.normal_(m.feedback.net.weight, std=0.3)
    with torch.no_grad():
        with feedback_mode(m, mode="loop"):
            out = forward_lm(m, x, opts={"holes": 3})
        Tn = int((out.cu1[1:] - out.cu1[:-1]).max())
        with feedback_mode(m, rounds=Tn):
            full = forward_lm(m, x, opts={"holes": 3}).logits
    assert torch.allclose(out.logits, full, atol=2e-4)
    out = model_forward(m, x, mc)
    out.logits.float().pow(2).mean().backward()
    assert float(m.feedback.net.weight.grad.abs().sum()) > 0 and float(m.feedback.norm.weight.grad.abs().sum()) > 0


def test_paper_exact_detached_passes():
    """detach (Latent Feedback's training, NOT end to end): same forward as the end-to-end passes, but the gradient
    reaches the main network and the feedback only through the last pass. affine False and init_std 0.02: no gain in
    the norm, W starts random."""
    m, mc = make(live=False, norm="rms", bias=False, affine=False, init_std=0.02, passes=2)
    assert m.feedback.norm.weight is None and 0.01 < float(m.feedback.net.weight.std()) < 0.03
    x = torch.tensor([list(b"\\xfeab cd \\x03 ef gh ij \\x03\\x03 kl mn op qr st uv wx")] * 2)
    with torch.no_grad():
        e2e = forward_lm(m, x, opts={"holes": 3}).logits
    m._feedback_cfg = dict(m._feedback_cfg, detach=True)
    with torch.no_grad():
        det = forward_lm(m, x, opts={"holes": 3}).logits
    assert torch.allclose(e2e, det, atol=1e-6)
    out = model_forward(m, x, mc)
    out.logits.float().pow(2).mean().backward()
    assert float(m.feedback.net.weight.grad.abs().sum()) > 0
    assert sum(float(p.grad.abs().sum()) for p in m.backbone.main_network.parameters() if p.grad is not None) > 0
