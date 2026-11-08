"""Build the calibration set: 2,048 samples that look like real serving traffic.

    python quantize/build_calib.py --model data/models/Qwen3.5-9B --out data/calib.jsonl

Needs the BF16 model served at --base-url, which writes the answers for the first two parts:

  800  reasoning traces: NuminaMath-1.5 (not MATH, to keep MATH-500 clean) and OpenCodeReasoning prompts,
       answered by the model with thinking on
  400  tool calls: hermes-function-calling-v1 prompts and tool schemas, answered through the native tool
       parser, half with thinking
  512  ultrachat_200k multi-turn chat, as is
  336  neuralmagic/calibration instruction, STEM and code, as is

Every sample is rendered with Qwen3.5's own chat template (<think> blocks, tool-call XML), shuffled, and
written as one JSON line with its text and token count. Statistics go to stdout.
"""

import argparse
import asyncio
import json
import random
import sys
from collections import Counter

from datasets import load_dataset
from openai import AsyncOpenAI
from transformers import AutoTokenizer

SEED = 42
# Dataset commits as of 2026-11-10; each one's last change predates the run.
REVISIONS = {
    "project-numina/NuminaMath-1.5": "1b05109f9e5c1ad06c0663519502416c30b300f8",
    "nvidia/OpenCodeReasoning": "20a1ca19c0d050fe9057fc08339d6b370ec1c67a",
    "NousResearch/hermes-function-calling-v1": "dae3e1d28cfbcf4b915c04ea1e072030529b4bda",
    "HuggingFaceH4/ultrachat_200k": "8049631c405ae6576f93f445c6b8166f76f5505a",
    "neuralmagic/calibration": "fb6bc2f8c66543876fb31613f5872b9030220e15",
}


def load(repo: str, *args, **kw):
    return load_dataset(repo, *args, revision=REVISIONS[repo], **kw)


def first(rows, n: int, keep) -> list:
    out = []
    for r in rows:
        if keep(r):
            out.append(r)
            if len(out) == n:
                return out
    return out


def math_prompts(n: int) -> list:
    rows = load("project-numina/NuminaMath-1.5", split="train", streaming=True).shuffle(seed=SEED, buffer_size=20000)
    rows = first(rows, n, lambda r: r["source"] != "math" and r["problem_is_valid"] == "Yes"
                 and r["question_type"] == "math-word-problem" and len(r["problem"]) < 2000)  # fmt: skip
    suffix = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    return [{"src": "numina", "messages": [{"role": "user", "content": r["problem"].strip() + suffix}]} for r in rows]


def code_prompts(n: int) -> list:
    rows = load("nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True)
    rows = first(rows.shuffle(seed=SEED, buffer_size=5000), n, lambda r: len(r["input"]) < 4000)
    suffix = "\n\nWrite a Python solution."
    return [{"src": "opencodereasoning", "messages": [{"role": "user", "content": r["input"].strip() + suffix}]}
            for r in rows]  # fmt: skip


def tool_prompts(n: int) -> list:
    rows = load("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", split="train")
    out = []
    for r in rows.shuffle(seed=SEED).select(range(n)):
        user = next(c["value"] for c in r["conversations"] if c["from"] == "human")
        tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
        out.append({"src": "hermes_fc", "messages": [{"role": "user", "content": user}], "tools": tools})
    return out


def chat_samples(repo: str, src: str, n: int, *args, split: str) -> list:
    rows = load(repo, *args, split=split).shuffle(seed=SEED).select(range(n))
    return [{"src": src, "messages": r["messages"], "thinking": False} for r in rows]


async def answer_all(base_url: str, jobs: list, concurrency: int) -> tuple[list, int]:
    """Let the served model answer every (sample, thinking, max_tokens) job in one batch; drop failures."""
    client = AsyncOpenAI(base_url=base_url, api_key="x", timeout=3600, max_retries=4)
    model = (await client.models.list()).data[0].id
    slots = asyncio.Semaphore(concurrency)

    async def answer(sample, thinking, max_tokens):
        req = {"model": model, "messages": sample["messages"], "max_tokens": max_tokens, "temperature": 0.6,
               "top_p": 0.95, "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking}}}  # fmt: skip
        if "tools" in sample:
            req["tools"] = sample["tools"]
        for _ in range(3):  # sampling is random, so a retry can also fix malformed tool-call arguments
            try:
                async with slots:
                    r = await client.chat.completions.create(**req)
                m = r.choices[0].message
                msg = {"role": "assistant", "content": m.content or ""}
                if reasoning := getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None):
                    msg["reasoning_content"] = reasoning
                if m.tool_calls:
                    calls = [(tc.function.name, json.loads(tc.function.arguments)) for tc in m.tool_calls]
                    if not all(isinstance(args, dict) for _, args in calls):  # the template renders a mapping
                        raise ValueError("tool-call arguments are not a JSON object")
                    msg["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": args}}
                                         for n, args in calls]  # fmt: skip
                return sample | {"messages": sample["messages"] + [msg], "thinking": thinking}
            except Exception as e:
                error = e
        print(f"dropped a {sample['src']} sample: {error!r:.200}", file=sys.stderr)
        return None

    out = await asyncio.gather(*(answer(*job) for job in jobs))
    return [s for s in out if s], out.count(None)


def main():
    ap = argparse.ArgumentParser(description="Build the calibration set.")
    ap.add_argument("--model", required=True, help="BF16 checkpoint directory (for its tokenizer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=8192, help="per generated answer, thinking included")
    a = ap.parse_args()

    reasoning = math_prompts(400) + code_prompts(400)
    tools = tool_prompts(400)
    jobs = [(s, True, a.max_tokens) for s in reasoning + tools[:200]] + [(s, False, 2048) for s in tools[200:]]
    generated, dropped = asyncio.run(answer_all(a.base_url, jobs, a.concurrency))
    samples = (
        generated
        + chat_samples("HuggingFaceH4/ultrachat_200k", "ultrachat", 512, split="train_sft")
        + chat_samples("neuralmagic/calibration", "nm_calibration", 336, "LLM", split="train")
    )
    random.Random(SEED).shuffle(samples)

    tok = AutoTokenizer.from_pretrained(a.model)
    counts, lengths = Counter(), []
    with open(a.out, "w") as f:
        for s in samples:
            text = tok.apply_chat_template(s["messages"], tools=s.get("tools"), tokenize=False,
                                           enable_thinking=s.get("thinking", False))  # fmt: skip
            n = len(tok(text, add_special_tokens=False).input_ids)
            counts[s["src"]] += 1
            lengths.append(n)
            f.write(json.dumps({"src": s["src"], "text": text, "n_tokens": n}) + "\n")
    lengths.sort()
    stats = {"counts": counts, "n": len(lengths), "dropped": dropped, "tokens_total": sum(lengths),
             "tokens_p50": lengths[len(lengths) // 2], "tokens_p90": lengths[int(len(lengths) * 0.9)],
             "frac_over_8192": round(sum(n > 8192 for n in lengths) / len(lengths), 3)}  # fmt: skip
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
