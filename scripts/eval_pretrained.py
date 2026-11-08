"""Sanity check of our data + eval pipeline: val BPB of the official pretrained H-Net (paper: ~0.755 on FineWeb-Edu).
usage: python scripts/eval_pretrained.py third_party/hnet/configs/hnet_1stage_L.json data/pretrained/hnet_1stage_L.pt"""
import json, sys
import latents  # noqa: F401
import torch
from latents.models import build_model
from latents.train import Data, evaluate

cfg = json.load(open(sys.argv[1]))
mc = dict(arch_layout=cfg["arch_layout"], d_model=cfg["d_model"], d_intermediate=cfg["d_intermediate"],
          vocab_size_padded=256, ssm_cfg=cfg["ssm_cfg"], num_heads=cfg["attn_cfg"]["num_heads"],
          rotary_emb_dim=cfg["attn_cfg"]["rotary_emb_dim"], window_size=cfg["attn_cfg"]["window_size"])
m = build_model(mc)
sd = torch.load(sys.argv[2], map_location="cuda")
print(m.load_state_dict(sd, strict=True))
m = m.bfloat16()
res = evaluate(m, Data("bytes", "data/fwe"), mc, dict(seq_len=8192, micro_batch=4))
print(json.dumps(res))
