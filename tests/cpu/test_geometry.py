"""scripts/latent_geometry.py end to end on a tiny trained latent model (CPU)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

import tests.cpu_stubs as S
from latents import tasks
from latents import train as T

ROOT = Path(__file__).resolve().parents[2]
DATA = "data/_cpu_geom"


def test_latent_geometry_runs_and_reports(tmp_path):
    tasks.write(ROOT / DATA, train_bytes=3e5, val_bytes=6e4, n_test=8, seed=0)
    name = "cpu-geom"
    shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
    mc = S.tiny_mc(arch_layout=["m1", ["T2"], "m1"], dec_byte_emb=True, pool_k=0, N=4, chunk_pool="mean", alpha=1.0)
    for k in ("vocab_size_padded", "vocab_size", "seq_len", "bytes_per_token"):
        mc.pop(k)
    lat = dict(target="ema", ema_tau=0.99, ema_tau_end=0.99, head="mlp", dist="l2", decode_from_pred=True,
               detach_main=True, detach_trunk=True, target_pool=True, reg="vicreg", reg_scope="within", reg_gamma=1.0,
               target_local=True, target_local_cap=64)
    cfg = dict(name=name, group="cpu-test", data_dir=DATA, wandb=False, upload_hf=False, model=mc, latent=lat,
               train=dict(seq_len=256, batch_size=4, micro_batch=2, steps=4, total_bytes=1e4, lr=1e-3, wd=0.1,
                          eval_every=2, eval_windows=2, final_eval_windows=2, log_every=1, seed=0))
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    try:
        T.main([str(p)])
        out = tmp_path / "geom.json"
        r = subprocess.run([sys.executable, "-c", "import tests.cpu_stubs as S; S.install(); import runpy, sys; "
                            f"sys.argv=['latent_geometry.py', '{out}', 'Tiny=runs/{name}', '--windows', '8', "
                            f"'--data', '{DATA}', '--plot', '{tmp_path / 'g.png'}'];"
                            "runpy.run_path('scripts/latent_geometry.py', run_name='__main__')"],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
        res = json.load(open(out))["Tiny"]
        for kind in ("web", "chains"):
            for k in ("spread_ratio", "r2_head", "r2_rescaled", "r2_copy", "r2_linear", "dirs_r2_gt_0.1", "predictable_share"):
                assert k in res[kind], (kind, k)
        assert "digit_r2_head" in res["chains"]
        assert "next_word_top10" in res["web"] and "examples" in res["web"]
        assert (tmp_path / "g.png").exists()
    finally:
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
        shutil.rmtree(ROOT / DATA, ignore_errors=True)
