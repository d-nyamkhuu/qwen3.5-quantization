"""Quantize Qwen3.5-9B with llm-compressor into a compressed-tensors checkpoint.

    python quantize/quantize.py --scheme nvfp4 --model data/models/Qwen3.5-9B \
        --calib data/calib.jsonl --out data/models/Qwen3.5-9B-NVFP4

Recipes, adapted from llm-compressor's official Qwen3.5 examples to the 9B's 32 layers:

  fp8    FP8 W8A8 on every text Linear (per-channel weights, dynamic per-token activations),
         static per-tensor FP8 KV-cache scales
  w4a16  AWQ -> GPTQ, INT4 weights (symmetric, group 128); BF16 activations and KV cache
  nvfp4  AWQ -> GPTQ; MLPs -> NVFP4 (FP4 values, FP8 scale per 16); attention, Gated DeltaNet projections,
         lm_head and the last 4 MLPs -> FP8 dynamic; static FP8 KV-cache scales

Every variant keeps the vision tower, embeddings, the 32-wide Gated DeltaNet gates (in_proj_a/b), conv1d, norms
and the MTP head in BF16 (lm_head too, except in nvfp4).
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4, QuantizationScheme  # noqa: E402
from datasets import Dataset  # noqa: E402
from llmcompressor import oneshot  # noqa: E402
from llmcompressor.modifiers.gptq import GPTQModifier  # noqa: E402
from llmcompressor.modifiers.quantization import QuantizationModifier  # noqa: E402
from llmcompressor.modifiers.transform.awq import AWQModifier  # noqa: E402
from llmcompressor.utils import load_context  # noqa: E402
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration  # noqa: E402

IGNORE = ["re:visual.*", "re:model.visual.*", "re:.*embed_tokens$", r"re:.*linear_attn\.in_proj_a$",
          r"re:.*linear_attn\.in_proj_b$"]  # fmt: skip
KV_FP8 = {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "tensor", "dynamic": False,
          "observer": "static_minmax"}  # fmt: skip
FP8_TARGETS = [
    r"re:.*self_attn\.(q|k|v|o)_proj$",
    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r"re:.*lm_head",
    r"re:.*layers\.(28|29|30|31)\.mlp\..*(gate|up|down)_proj$",  # the last 4 of 32 layers stay FP8
]


def recipe(scheme: str, awq_offload) -> list:
    if scheme == "fp8":
        return [QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE + [r"re:.*lm_head"],
                                     kv_cache_scheme=KV_FP8)]  # fmt: skip
    awq = AWQModifier(duo_scaling="both", offload_device=awq_offload)
    if scheme == "w4a16":
        return [awq, GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE + [r"re:.*lm_head"])]
    groups = {
        "fp8": QuantizationScheme(targets=FP8_TARGETS, **FP8_DYNAMIC),
        "nvfp4": QuantizationScheme(targets=[r"re:.*mlp\..*(gate|up|down)_proj$"], **NVFP4),
    }
    return [awq, GPTQModifier(config_groups=groups, ignore=IGNORE, kv_cache_scheme=KV_FP8)]


def main():
    ap = argparse.ArgumentParser(description="Quantize Qwen3.5-9B with llm-compressor.")
    ap.add_argument("--scheme", choices=["fp8", "w4a16", "nvfp4"], required=True)
    ap.add_argument("--model", required=True, help="BF16 checkpoint directory")
    ap.add_argument("--calib", required=True, help="calibration JSONL from build_calib.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=256, help="first N calibration samples (512 for fp8)")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--awq-offload", choices=["gpu", "cpu"], default="gpu",
                    help="where AWQ caches block inputs: cpu is needed beyond ~256 samples on 96 GB, ~35%% slower")  # fmt: skip
    a = ap.parse_args()

    t0 = time.time()
    with load_context(Qwen3_5ForConditionalGeneration, load_mtp=False):  # the MTP head is copied unchanged
        model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype="auto")
    processor = AutoProcessor.from_pretrained(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)

    texts = [json.loads(line)["text"] for line in open(a.calib)][: a.num_samples]
    ds = Dataset.from_dict({"text": texts}).map(
        lambda s: tok(s["text"], max_length=a.max_seq_len, truncation=True, add_special_tokens=False),
        remove_columns=["text"],
    )
    offload = torch.device("cpu") if a.awq_offload == "cpu" else None
    oneshot(model=model, processor=processor, dataset=ds, recipe=recipe(a.scheme, offload),
            max_seq_length=a.max_seq_len, num_calibration_samples=len(texts))  # fmt: skip

    model.save_pretrained(a.out, save_compressed=True)
    processor.save_pretrained(a.out)
    meta = {"scheme": a.scheme, "num_samples": len(texts), "max_seq_len": a.max_seq_len,
            "awq_offload": a.awq_offload, "minutes": round((time.time() - t0) / 60, 1)}  # fmt: skip
    Path(a.out, "quantize_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
