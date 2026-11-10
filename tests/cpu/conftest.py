"""CPU tests: the official H-Net modules run with pure-PyTorch kernel stubs (tests/cpu_stubs.py).
usage: HNET_SRC=<goombalab/hnet checkout> python -m pytest tests/cpu -q   (default HNET_SRC: third_party/hnet)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tests.cpu_stubs as S  # noqa: E402

S.install()
import latents  # noqa: E402,F401
from torch.nn.attention import SDPBackend  # noqa: E402
import latents.rollout  # noqa: E402

latents.rollout.sdpa_kernel_backend = SDPBackend.MATH  # the mem-efficient kernel is CUDA-only
