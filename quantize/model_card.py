"""Write the Hugging Face model card (README.md) into a quantized checkpoint directory.

    python quantize/model_card.py nvfp4 data/models/Qwen3.5-9B-NVFP4 --repo nyamkhuu/Qwen3.5-9B-NVFP4-AWQ-GPTQ

Results come from results/summary.json.
"""

import argparse
import json
from pathlib import Path

GITHUB = "https://github.com/d-nyamkhuu/qwen3.5-quantization"
ENGINES = {"vllm": "vLLM", "sglang": "SGLang"}
RECIPES = {
    "fp8": ("FP8 (W8A8) + FP8 KV cache", "fp8", 512,
            "FP8 E4M3 weights (per-channel scales) and dynamic per-token FP8 activations on every text Linear, "
            "plus static per-tensor FP8 KV-cache scales for the 8 full-attention layers."),
    "w4a16": ("W4A16 INT4 (AWQ + GPTQ)", "int4", 256,
              "AWQ (`duo_scaling=\"both\"`) then GPTQ: INT4 symmetric weights in groups of 128, BF16 activations "
              "and KV cache."),
    "nvfp4": ("NVFP4 (W4A4) + FP8 (AWQ + GPTQ)", "nvfp4", 256,
              "AWQ then GPTQ with two groups: the MLPs of layers 0–27 in **NVFP4** (FP4 values, an FP8 scale per "
              "16 values); attention, Gated DeltaNet projections, `lm_head` and the MLPs of layers 28–31 in FP8 "
              "dynamic; static per-tensor FP8 KV-cache scales."),
}  # fmt: skip

CARD = """---
base_model: Qwen/Qwen3.5-9B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags: [qwen3.5, compressed-tensors, llm-compressor, {tag}, vllm, sglang]
---

# Qwen3.5-9B {title}

[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) quantized with
[llm-compressor](https://github.com/vllm-project/llm-compressor) 0.15.0 into compressed-tensors format, then
served, benchmarked and evaluated on vLLM 0.33 and SGLang 0.5.23 on one NVIDIA RTX PRO 6000 Blackwell.
Code, method and full results: [{github}]({github}).

## Recipe

{recipe} The vision tower, embeddings, the 32-wide Gated DeltaNet gates, `conv1d`, norms and the MTP head stay
in BF16. Calibrated on {samples} samples (up to 8,192 tokens) of a 2,048-sample set of the model's own reasoning
traces and tool calls, multi-turn chat and instruction data, rendered with its chat template.

## Serving

```bash
vllm serve {repo} --language-model-only --max-model-len 32768 \\
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

```bash
python -m sglang.launch_server --model-path {repo} --context-length 32768 \\
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

## Results vs BF16 (one RTX PRO 6000)

| Engine | Variant | Single-user tokens/s | Peak tokens/s | KV-cache tokens | MATH-100 | GPQA-D-100 | BFCL | JSON exact |
|---|---|---|---|---|---|---|---|---|
{rows}

1,024 tokens in, 512 out. MATH and GPQA use thinking capped at 16,384 tokens (±5 points standard error at n=100);
BFCL v3 non-live has 1,000 questions; JSON is NousResearch/json-mode-eval.
"""


def main():
    ap = argparse.ArgumentParser(description="Write a model card into a checkpoint directory.")
    ap.add_argument("variant", choices=RECIPES)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--repo", required=True, help="Hugging Face repo id")
    ap.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    a = ap.parse_args()
    runs = json.loads(a.summary.read_text())["runs"]
    title, tag, samples, recipe = RECIPES[a.variant]

    rows = []
    for engine in ("vllm", "sglang"):
        for variant in (a.variant, "bf16"):
            r = runs[f"{variant}/{engine}"]
            e = r["evals"]
            rows.append(
                f"| {ENGINES[engine]} | {variant.upper()} | {r['perf']['1']['output_throughput']:.0f} | "
                f"{r['peak']['output_tok_s']:,.0f} | {r['memory']['kv_tokens']:,} | {e['math']['acc']:.0%} | "
                f"{e['gpqa']['acc']:.0%} | {e['bfcl']['acc']:.1%} | {e['json']['correct']:.0%} |"
            )
    card = CARD.format(title=title, tag=tag, samples=samples, recipe=recipe, repo=a.repo, github=GITHUB,
                       rows="\n".join(rows))  # fmt: skip
    (a.dir / "README.md").write_text(card)
    print(f"wrote {a.dir / 'README.md'}")


if __name__ == "__main__":
    main()
