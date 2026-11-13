"""H-Net latents project.

Compat shim: mamba_ssm 2.2.5 (the newest with a prebuilt torch-2.8 wheel) imports a few `transformers`
names only for its own HF-loading / generation utilities, which we never use. Recent transformers removed
them and an old transformers would force-downgrade huggingface_hub/tokenizers, so if transformers is not
importable we register inert stand-ins.
"""
import sys
import types
from collections import namedtuple

try:  # pragma: no cover
    from transformers.generation import GreedySearchDecoderOnlyOutput  # noqa: F401
except Exception:
    def _mod(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    # `datasets` checks pickled objects against transformers.PreTrainedTokenizerBase when transformers is imported
    _mod("transformers", PreTrainedTokenizerBase=type("PreTrainedTokenizerBase", (), {}))
    _mod("transformers.generation",
         GreedySearchDecoderOnlyOutput=namedtuple("GreedySearchDecoderOnlyOutput", ["sequences", "scores"]),
         SampleDecoderOnlyOutput=namedtuple("SampleDecoderOnlyOutput", ["sequences", "scores"]),
         TextStreamer=object)
    _mod("transformers.utils", WEIGHTS_NAME="pytorch_model.bin", CONFIG_NAME="config.json")
    _mod("transformers.utils.hub", cached_file=None)
