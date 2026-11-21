"""Next-latent prediction on a byte H-Net (latents/nextlat.py): rollout indexing, where the gradient goes, the KL
term's decoder pass, and the trainer end to end."""
import json
import shutil
from pathlib import Path

import pytest
import torch

import tests.cpu_stubs as S
from latents import tasks, train as T
from latents.models import build_model, forward_lm, param_groups

ROOT = Path(__file__).resolve().parents[2]
DATA = "data/_cpu_task_nl"


def make(seed=0, **nl):
    torch.manual_seed(seed)
    mc = S.tiny_mc(seq_len=96, nextlat=dict(dict(weight=1.0, hidden=48), **nl))
    m = build_model(mc, device="cpu")
    torch.nn.init.normal_(m.nextlat.mlp[-1].weight, std=0.05)  # away from the zero init, so every part gets gradient
    x = torch.randint(0, 256, (3, 96))
    return m, mc, x


def test_rollout_stays_inside_a_row_and_copy_at_init():
    m, mc, x = make(horizon=3)
    out = forward_lm(m, x)
    cu1 = out.cu1.long()
    lens = (cu1[1:] - cu1[:-1]).tolist()
    res = m.nextlat.rollout(out)
    for i, (idx, pred) in enumerate(res, start=1):
        assert idx.numel() == sum(max(n - i, 0) for n in lens) and pred.shape == (idx.numel(), mc["d_model"][-1])
        row = torch.searchsorted(cu1[1:], idx, right=True)
        assert (idx - i >= cu1[:-1][row]).all()  # the start chunk is in the target's own row
    torch.nn.init.zeros_(m.nextlat.mlp[-1].weight)  # as built: the prediction is the current state
    idx, pred = m.nextlat.rollout(out)[0]
    assert torch.allclose(pred, out.z_full[idx - 1].float(), atol=1e-6)
    em = m.nextlat.eval_metrics(out)
    assert abs(em["nl/explained"] - em["nl/explained_copy"]) < 1e-5 and em["nl/state_rank"] > 1


def test_gradient_reaches_main_encoder_and_the_transition_model_not_the_decoder():
    m, mc, x = make()
    out = forward_lm(m, x)
    loss, st = m.nextlat.loss(m, out, x, mc)
    assert torch.isfinite(loss) and float(st["nl_loss"]) > 0
    loss.backward()
    bb = m.backbone

    def gn(mod):
        return sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)
    assert gn(bb.main_network) > 0 and gn(bb.encoder) > 0 and gn(m.nextlat) > 0
    assert gn(bb.decoder) == 0 and gn(m.lm_head) == 0
    # the target is not a path for the gradient: with the prediction cut off from the model, nothing reaches it
    m.zero_grad()
    out = forward_lm(m, x)
    idx, _ = m.nextlat.rollout(out)[0]
    pred = m.nextlat.step(out.z_full[idx - 1].detach(), out.chunks[idx].detach())
    torch.nn.functional.smooth_l1_loss(pred, out.z_full.detach().float()[idx]).backward()
    assert gn(bb.main_network) == 0 and gn(bb.encoder) == 0 and gn(m.nextlat) > 0


def test_kl_pass_equals_the_normal_decoder_and_leaves_it_untrained():
    m, mc, x = make(kl=1.0)
    out = forward_lm(m, x)
    nl = m.nextlat
    idx = nl.rollout(out)[0][0]
    # fed the real states, the second decoder pass reproduces the normal byte distribution: KL = 0
    assert float(nl.kl_loss(m, out, x, mc, [(idx, out.z_full[idx].float())])) < 1e-6
    loss, st = nl.loss(m, out, x, mc)
    assert float(st["nl_kl"]) > 0
    kl = nl.kl_loss(m, out, x, mc, nl.rollout(out))
    kl.backward()
    bb = m.backbone
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for mod in (bb.decoder, m.lm_head, bb.residual_proj)
               for p in mod.parameters())
    assert all(p.requires_grad for p in bb.decoder.parameters())
    assert sum(float(p.grad.abs().sum()) for p in nl.parameters() if p.grad is not None) > 0
    assert sum(float(p.grad.abs().sum()) for p in bb.main_network.parameters() if p.grad is not None) > 0


def test_optimizer_sees_the_transition_model():
    m, mc, _ = make()
    groups, mults = param_groups(m, mc, 0.1)
    names = [n for g in groups for n in g["names"]]
    assert any(n.startswith("nextlat.") for n in names)
    g = next(g for g in groups if any(n.startswith("nextlat.mlp.0.weight") for n in g["names"]))
    assert g["lr_mult"] == mults[-1]


@pytest.mark.parametrize("nl", [dict(weight=1.0, hidden=48), dict(weight=1.0, kl=1.0, horizon=2, hidden=48)])
def test_trainer_runs_with_nextlat(tmp_path, nl):
    import yaml
    tasks.write(ROOT / DATA, train_bytes=3e5, val_bytes=2e4, n_test=12, seed=0)
    name = "cpu-nextlat"
    shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
    mc = S.tiny_mc(nextlat=nl)
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
        assert sm["val/bpb"] > 0 and "val/nl/explained" in sm and "val/nl/explained_copy" in sm
        ck = torch.load(ROOT / "runs" / name / "final.pt", weights_only=False)
        build_model(ck["cfg"]["model"], device="cpu").load_state_dict(ck["model"], strict=True)
        rows = [json.loads(l) for l in open(ROOT / "runs" / name / "metrics.jsonl")]
        assert any("train/nl_loss" in r for r in rows)
    finally:
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
        shutil.rmtree(ROOT / DATA, ignore_errors=True)


@pytest.mark.parametrize("norm", ["batch", "scale"])
def test_normed_loss_ignores_the_scale_of_the_variation(norm):
    """With norm=batch / scale, shrinking the states' variation around their mean does not lower the loss."""
    m, mc, x = make(norm=norm)
    out = forward_lm(m, x)
    l1, _ = m.nextlat.loss(m, out, x, mc)
    mu = out.z_full.mean(0, keepdim=True)
    out2 = forward_lm(m, x)
    out2.z_full = mu + 0.1 * (out.z_full - mu)
    torch.nn.init.zeros_(m.nextlat.mlp[-1].weight)  # copy prediction: the same error in units of the spread
    l_small, _ = m.nextlat.loss(m, out2, x, mc)
    l_big, _ = m.nextlat.loss(m, out, x, mc)
    assert abs(float(l_small) - float(l_big)) < 1e-3 * float(l_big)
    m2, mc2, _ = make(norm="none")
    torch.nn.init.zeros_(m2.nextlat.mlp[-1].weight)
    o = forward_lm(m2, x)
    o2 = forward_lm(m2, x)
    o2.z_full = o.z_full.mean(0, keepdim=True) + 0.1 * (o.z_full - o.z_full.mean(0, keepdim=True))
    assert float(m2.nextlat.loss(m2, o2, x, mc2)[0]) < 0.1 * float(m2.nextlat.loss(m2, o, x, mc2)[0])
