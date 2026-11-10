"""Accuracy evals against any OpenAI-compatible server (vLLM or SGLang).

    python evals/run_evals.py --out results/bench/q35_9b_nvfp4_vllm/evals

Suites (all run at once through one pool of request slots, so the server batch stays full):
  math              MATH-500, 100 questions in seeded order, thinking on, graded with math-verify
  gpqa              GPQA-Diamond, first 100 questions, thinking on, answer letter in \\boxed{}
  bfcl              BFCL v3 non-live simple / multiple / parallel / parallel_multiple (1,000), native tool
                    calls through the engine's parser, matched against the possible answers
  json              NousResearch/json-mode-eval (100): parses, matches the schema, equals the reference
  json_constrained  the same with the engine enforcing the schema (response_format=json_schema)

Writes every response to <out>/<suite>.jsonl and the scores to <out>/evals_summary.json. A request that
still fails after the client's retries is graded wrong; `infra_errors` counts failures that are not a 4xx
rejection of that request (dead server, timeout), and the script exits 1 if any suite has one.
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx2 as httpx  # the httpx fork openai>=3 uses; its client sets the connection limit
import jsonschema
import pandas as pd
from datasets import load_dataset
from math_verify import parse, verify
from openai import APIStatusError, AsyncOpenAI

BFCL_URL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
BFCL_CATEGORIES = ["simple", "multiple", "parallel", "parallel_multiple"]
GPQA_URL = "https://huggingface.co/datasets/fingertap/GPQA-Diamond/resolve/main/test/gpqa_diamond.parquet"
JSON_URL = "https://huggingface.co/datasets/NousResearch/json-mode-eval/resolve/main/data/train-00000-of-00001.parquet"
BOXED = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

# Qwen3.5 model-card sampling for thinking mode; greedy without thinking for tools and JSON
THINKING = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k": 20, "min_p": 0.0}}
GREEDY = {"temperature": 0.0, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


# ------------------------------------------------------------------ suites: (items, request, grade)
def math_suite(a):
    rows = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    random.Random(a.seed).shuffle(rows)
    items = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]} for r in rows[: a.limit_math]]

    def request(it):
        return THINKING | {"messages": [{"role": "user", "content": it["problem"] + BOXED}],
                           "max_tokens": a.think_max_tokens, "seed": a.seed}  # fmt: skip

    def grade(it, rec):
        try:
            pred = parse(rec["content"] or "")
            return {"correct": bool(pred) and verify(parse(f"${it['answer']}$"), pred), "gold": it["answer"]}
        except Exception:  # math-verify raises on unparsable answers
            return {"correct": False, "gold": it["answer"]}

    return items, request, grade


def gpqa_suite(a):
    df = pd.read_parquet(GPQA_URL)
    items = [{"id": f"gpqa_{i:03d}", "question": r["question"], "answer": str(r["answer"]).strip()}
             for i, r in df.iterrows()][: a.limit_gpqa]  # fmt: skip
    prompt = "\n\nAnswer with the letter of the correct option. Put the final letter within \\boxed{}."

    def request(it):
        return THINKING | {"messages": [{"role": "user", "content": it["question"] + prompt}],
                           "max_tokens": a.think_max_tokens, "seed": a.seed}  # fmt: skip

    def grade(it, rec):
        text = rec["content"] or ""
        letters = re.findall(r"\\boxed\{\s*\(?([A-D])\)?\s*\}", text) or re.findall(
            r"[Aa]nswer\s*[:is]*\s*\(?([A-D])\)?", text
        )
        pred = letters[-1] if letters else None
        return {"correct": pred == it["answer"][:1].upper(), "pred": pred, "gold": it["answer"]}

    return items, request, grade


BFCL_TYPES = {"dict": "object", "float": "number", "tuple": "array", "int": "integer", "list": "array",
              "bool": "boolean", "str": "string", "any": None}  # fmt: skip


def json_schema(p):
    """BFCL's Python-flavoured parameter types -> JSON Schema."""
    if isinstance(p, list):
        return [json_schema(x) for x in p]
    if not isinstance(p, dict):
        return p
    out = {}
    for k, v in p.items():
        if k == "type" and isinstance(v, str):
            if (t := BFCL_TYPES.get(v, v)) is not None:
                out["type"] = t
        elif k == "properties":
            out[k] = {name: json_schema(s) for name, s in v.items()}
        else:
            out[k] = json_schema(v)
    return out


def tool_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def value_ok(v, acceptable: list) -> bool:
    """Whether one argument value matches any acceptable value (strings compared without punctuation/case)."""
    norm = lambda s: re.sub(r"[ ,./\-_*^]", "", str(s)).lower()  # noqa: E731
    for want in acceptable:
        if want in ("", None):
            continue
        if isinstance(want, bool) or isinstance(v, bool):
            ok = v is want or (isinstance(v, str) and v.lower() == str(want).lower())
        elif isinstance(want, (int, float)):
            ok = isinstance(v, (int, float, str)) and _float(v) == float(want)
        elif isinstance(want, str):
            ok = isinstance(v, str) and norm(v) == norm(want)
        elif isinstance(want, list):
            ok = isinstance(v, (list, tuple)) and len(v) == len(want) and all(value_ok(x, [y]) for x, y in zip(v, want))
        elif isinstance(want, dict):
            ok = isinstance(v, dict) and args_ok(v, want)
        else:
            ok = False
        if ok:
            return True
    return False


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def args_ok(args: dict, spec: dict) -> bool:
    """spec maps each parameter to its acceptable values; "" among them marks it optional."""
    if any(k not in spec for k in args):
        return False
    for k, acceptable in spec.items():
        acceptable = acceptable if isinstance(acceptable, list) else [acceptable]
        if k not in args and "" not in acceptable and None not in acceptable:
            return False
        if k in args and not value_ok(args[k], acceptable):
            return False
    return True


def calls_ok(calls: list, expected: list, names: dict) -> bool:
    """One-to-one, order-insensitive match of (name, args) calls against [{function: spec}, ...]."""
    if len(calls) != len(expected):
        return False
    if not expected:
        return True
    (fname, spec), = expected[0].items()  # fmt: skip
    return any(
        names.get(n, n) == fname and args_ok(a, spec) and calls_ok(calls[:i] + calls[i + 1 :], expected[1:], names)
        for i, (n, a) in enumerate(calls)
    )


def bfcl_suite(a):
    cache = Path(a.cache) / "bfcl"
    cache.mkdir(parents=True, exist_ok=True)
    items = []
    for cat in BFCL_CATEGORIES:
        files = {"q": (cache / f"{cat}.json", f"{BFCL_URL}/BFCL_v3_{cat}.json"),
                 "a": (cache / f"pa_{cat}.json", f"{BFCL_URL}/possible_answer/BFCL_v3_{cat}.json")}  # fmt: skip
        for path, url in files.values():
            if not path.exists():
                path.write_bytes(httpx.get(url, follow_redirects=True, timeout=60).content)
        answers = {(r := json.loads(line))["id"]: r["ground_truth"] for line in open(files["a"][0])}
        for line in open(files["q"][0]):
            r = json.loads(line)
            items.append({"id": r["id"], "cat": cat, "messages": r["question"][0], "functions": r["function"],
                          "gt": answers[r["id"]]})  # fmt: skip

    def request(it):
        tools = [{"type": "function", "function": {"name": tool_name(f["name"]), "description": f.get("description", ""),
                  "parameters": json_schema(f["parameters"])}} for f in it["functions"]]  # fmt: skip
        return GREEDY | {"messages": it["messages"], "tools": tools, "tool_choice": "auto", "max_tokens": 2048}

    def grade(it, rec):
        calls, parse_error = [], False
        for tc in rec["tool_calls"]:
            args = tc["function"]["arguments"]
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                args = None
            if isinstance(args, dict):
                calls.append((tc["function"]["name"], args))
            else:
                parse_error = True
        names = {tool_name(f["name"]): f["name"] for f in it["functions"]}
        return {"correct": not parse_error and calls_ok(calls, it["gt"], names), "cat": it["cat"],
                "tool_parse_error": parse_error, "emitted_any_call": bool(calls)}  # fmt: skip

    return items, request, grade


def json_suite(a, constrained: bool):
    df = pd.read_parquet(JSON_URL)
    items = [{"id": f"json_{i:03d}", "messages": [dict(m) for m in r["prompt"]], "schema": json.loads(r["schema"]),
              "ref": json.loads(r["completion"])} for i, r in df.iterrows()]  # fmt: skip

    def request(it):
        req = GREEDY | {"messages": it["messages"], "max_tokens": 2048}
        if constrained:
            req["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": it["schema"], "strict": True},
            }
        return req

    def grade(it, rec):
        text = (rec["content"] or "").strip()
        if m := re.search(r"```(?:json)?\s*(.*?)```", text, re.S):
            text = m[1].strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return {"parsed": False, "schema_valid": False, "correct": False}
        try:
            jsonschema.validate(obj, it["schema"])
            valid = True
        except (jsonschema.ValidationError, jsonschema.SchemaError):
            valid = False
        return {"parsed": True, "schema_valid": valid, "correct": obj == it["ref"]}

    return items, request, grade


SUITES = {
    "gpqa": (gpqa_suite, ["correct"]),
    "math": (math_suite, ["correct"]),
    "bfcl": (bfcl_suite, ["correct", "tool_parse_error", "emitted_any_call"]),
    "json": (lambda a: json_suite(a, False), ["parsed", "schema_valid", "correct"]),
    "json_constrained": (lambda a: json_suite(a, True), ["parsed", "schema_valid", "correct"]),
}


# ------------------------------------------------------------------ runner
async def ask(client, model, req: dict) -> dict:
    try:
        r = await client.chat.completions.create(model=model, **req)
    except Exception as e:  # graded wrong; a 4xx rejection of this one request is a valid result, the rest is not
        rejected = isinstance(e, APIStatusError) and 400 <= e.status_code < 500 and e.status_code not in (408, 409, 429)
        return {"content": None, "tool_calls": [], "finish_reason": "error", "completion_tokens": None,
                "error": repr(e)[:500], "error_type": "rejected" if rejected else "infra"}  # fmt: skip
    msg = r.choices[0].message
    return {
        "content": msg.content,
        "reasoning": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
        "tool_calls": [tc.model_dump() for tc in msg.tool_calls or []],
        "finish_reason": r.choices[0].finish_reason,
        "completion_tokens": r.usage.completion_tokens,
    }


def summarize(results: list, keys: list) -> dict:
    n = len(results)
    errors = [r for r in results if r["finish_reason"] == "error"]
    s = {
        "n": n,
        "errors": len(errors),
        "infra_errors": sum(r["error_type"] == "infra" for r in errors),
        "truncated": sum(r["finish_reason"] == "length" for r in results),
        "mean_completion_tokens": round(sum(r["completion_tokens"] or 0 for r in results) / n, 1),
    }
    return s | {k: round(sum(bool(r.get(k)) for r in results) / n, 4) for k in keys}


async def main():
    ap = argparse.ArgumentParser(description="Accuracy evals against an OpenAI-compatible server.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--cache", default="data/evalcache", help="where BFCL's files are downloaded")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit-math", type=int, default=100)
    ap.add_argument("--limit-gpqa", type=int, default=100)
    ap.add_argument("--think-max-tokens", type=int, default=16384)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=a.base_url, api_key="x", timeout=3600, max_retries=5,
                         http_client=httpx.AsyncClient(limits=httpx.Limits(max_connections=512), timeout=3600))  # fmt: skip
    model = (await client.models.list()).data[0].id
    slots = asyncio.Semaphore(a.concurrency)
    summary_path = a.out / "evals_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    async def run(suite: str):
        t0 = time.time()
        make, keys = SUITES[suite]
        items, request, grade = make(a)

        async def one(it):
            async with slots:
                rec = await ask(client, model, request(it))
            return {"id": it["id"]} | rec | grade(it, rec)

        results = sorted(await asyncio.gather(*map(one, items)), key=lambda r: str(r["id"]))
        (a.out / f"{suite}.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in results))
        s = summarize(results, keys)
        if suite == "bfcl":
            s["by_category"] = {c: summarize([r for r in results if r["cat"] == c], ["correct"])["correct"]
                                for c in BFCL_CATEGORIES}  # fmt: skip
        summary[suite] = s | {"wall_s": round(time.time() - t0, 1)}
        summary_path.write_text(json.dumps(summary, indent=2))
        print(suite, json.dumps(summary[suite]), flush=True)

    suites = [s for s in SUITES if s in a.suites.split(",")]  # longest first, so they get slots first
    await asyncio.gather(*map(run, suites))
    failed = [s for s in suites if summary[s]["infra_errors"]]
    if failed:
        sys.exit(f"infra errors in {', '.join(failed)}: those scores are not valid, rerun them with --suites")


if __name__ == "__main__":
    asyncio.run(main())
