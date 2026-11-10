"""Entry point for the CPU data-parallel test: install the kernel stubs, then run the trainer (gloo backend)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tests.cpu_stubs as S  # noqa: E402

S.install()
from torch.nn.attention import SDPBackend  # noqa: E402
import latents.rollout  # noqa: E402
from latents import train  # noqa: E402

latents.rollout.sdpa_kernel_backend = SDPBackend.MATH
train.main(sys.argv[1:])
