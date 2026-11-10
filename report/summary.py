"""Raw run outputs -> results/summary.json, with optional client and scheduler traces.

    python report/summary.py --results results --out results/summary.json

Reads, per benchmark run `bench/q35_9b_<variant>_<engine>/`: `perf/c<N>.json` (vllm bench serve),
`perf/gpu_samples.csv` + `perf/phases.txt` (nvidia-smi), `server.log` (memory) and `evals/evals_summary.json`;
Optional, per trace run `bench/*_trace*/`: detailed `perf/c128.json` for client token gaps,
plus `sched_trace/*.jsonl` for engine step durations. Missing traces are omitted.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

USD_PER_GPU_HOUR = 1.80  # on-demand RTX PRO 6000 price, November 2026
PERF_KEYS = ["output_throughput", "total_token_throughput"] + [
    f"{s}_{m}_ms" for m in ("ttft", "tpot", "itl", "e2el") for s in ("mean", "p50", "p95", "p99")
]
TRACES = {
    "vllm": "q35_9b_nvfp4_vllm_trace",
    "vllm_2048": "q35_9b_nvfp4_vllm_trace_mnbt2048",
    "sglang": "q35_9b_nvfp4_sglang_trace",
}


def perf(run: Path) -> dict:
    out = {}
    for f in (run / "perf").glob("c*.json"):
        r = json.loads(f.read_text())
        out[int(f.stem[1:])] = {k: round(r[k], 2) for k in PERF_KEYS}
    return dict(sorted(out.items()))


def gpu(run: Path) -> tuple[float, dict]:
    """Peak device memory (GiB) and mean power (W) per concurrency phase, skipping each phase's first 30 s."""
    rows = []
    for t, mem, _, watts in csv.reader(open(run / "perf/gpu_samples.csv")):
        ts = datetime.strptime(t.strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        rows.append((ts.timestamp(), float(mem), float(watts)))
    phases = {}
    for line in open(run / "perf/phases.txt"):
        ts, event, c = line.split()
        phases.setdefault(int(c[2:]), {})[event] = float(ts)
    power = {
        c: round(statistics.mean(w for t, _, w in rows if p["start"] + 30 <= t <= p["end"]), 1)
        for c, p in phases.items()
    }
    return round(max(m for _, m, _ in rows) / 1024, 1), power


def memory(log: str) -> dict:
    """Weights, KV capacity and state pools, as each engine logs them."""
    patterns = {
        "weights_gib": r"Model loading took ([\d.]+) GiB|Load weight end.*?mem usage=([\d.]+) GB",
        "kv_tokens": r"GPU KV cache size: ([\d,]+) tokens|KV Cache is allocated.*?#tokens: (\d+)",
        "kv_pool_gib": r"Available KV cache memory: ([\d.]+) GiB",
        "kv_gb": r"KV Cache is allocated.*?K size: ([\d.]+) GB, V size: ([\d.]+) GB",
        "gdn_gb": r"Mamba Cache is allocated.*?conv_state size: ([\d.]+)GB, ssm_state size: ([\d.]+)GB",
        "max_running_requests": r"max_running_requests=(\d+)",
    }
    out = {}
    for key, pattern in patterns.items():
        if m := re.search(pattern, log):
            values = [float(g.replace(",", "")) for g in m.groups() if g]
            out[key] = round(sum(values), 2)
    for key in ("kv_tokens", "max_running_requests"):
        if key in out:
            out[key] = int(out[key])
    return out


def evals(summary: dict) -> dict:
    out = {s: {"acc": summary[s]["correct"], "truncated": summary[s]["truncated"]} for s in ("math", "gpqa")}
    out["bfcl"] = {"acc": summary["bfcl"]["correct"], **summary["bfcl"]["by_category"]}
    for s in ("json", "json_constrained"):
        out[s] = {k: summary[s][k] for k in ("parsed", "schema_valid", "correct")}
    return out


def run_summary(run: Path) -> dict:
    p = perf(run)
    peak_c = max(p, key=lambda c: p[c]["output_throughput"])
    peak_tok_s = p[peak_c]["output_throughput"]
    peak_mem, power = gpu(run)
    return {
        "perf": p,
        "peak": {
            "concurrency": peak_c,
            "output_tok_s": peak_tok_s,
            "usd_per_1m_output_tok": round(USD_PER_GPU_HOUR / (peak_tok_s * 3600) * 1e6, 4),
            "power_w": power[peak_c],
            "output_tok_per_joule": round(peak_tok_s / power[peak_c], 2),
        },
        "memory": {**memory((run / "server.log").read_text(errors="ignore")), "peak_device_gib": peak_mem},
        "evals": evals(json.loads((run / "evals/evals_summary.json").read_text())),
    }


def pct(xs: list, p: float) -> float:
    xs = sorted(xs)
    return xs[round(p / 100 * (len(xs) - 1))]


def trace_summary(run: Path) -> dict | None:
    """Summarize available client token gaps, adding engine step metrics when recorded."""
    client = run / "perf/c128.json"
    if not client.exists():
        print(f"skipping trace {run.name}: no {client}", file=sys.stderr)
        return None
    r = json.loads(client.read_text())
    gaps = [[g * 1000 for g in req] for req in (r.get("itls") or [])]
    flat = [g for req in gaps for g in req]
    if not flat:
        print(f"skipping trace {run.name}: no detailed client token gaps", file=sys.stderr)
        return None
    worst = sorted(max(req, default=0.0) for req in gaps)
    summary = {
        "output_tok_s": round(r["output_throughput"]),
        "p99_ttft_s": round(r["p99_ttft_ms"] / 1000, 1),
        "itl_ms": {
            "mean": round(statistics.mean(flat), 1),
            "p99": round(pct(flat, 99)),
            "p99.9": round(pct(flat, 99.9)),
            "max": round(max(flat)),
        },
        "requests_with_gap_over_100ms": sum(w > 100 for w in worst),
        "worst_gap_per_request_ms": {"median": round(pct(worst, 50)), "all": [round(w) for w in worst]},
        # the token gaps of the request in the middle of the run, for the timeline chart
        "example_request_gaps_ms": [round(g, 1) for g in gaps[len(gaps) // 2]],
    }
    events = []
    for path in (run / "sched_trace").glob("*.jsonl"):
        with path.open() as source:
            events.extend(json.loads(line) for line in source)
    events.sort(key=lambda e: e["t"])
    if any(e["ev"] == "sched" for e in events):  # vLLM: a step ends at the next update_from_output
        sched = [e for e in events if e["ev"] == "sched" and e["tok"] > 0]
        outs = [e["t"] for e in events if e["ev"] == "out"]
        steps = [((outs[k] - outs[k - 1]) * 1000, sched[k]["pre_tok"]) for k in range(1, min(len(sched), len(outs)))]
    else:  # SGLang: a step ends at the next run_batch
        runs = [e for e in events if e["ev"] == "run"]
        steps = [
            ((b["t"] - a["t"]) * 1000, a["ext_tok"] if a["mode"] != "DECODE" else 0) for a, b in zip(runs, runs[1:])
        ]
    steps = [(d, pre) for d, pre in steps if d < 1000]  # drop idle gaps between benchmark phases
    decode = [d for d, pre in steps if pre == 0]
    prefill = [d for d, pre in steps if pre > 0]
    if steps:
        summary["step_ms"] = {
            "decode_only": round(statistics.median(decode)) if decode else None,
            "with_prefill": round(statistics.median(prefill)) if prefill else None,
        }
        summary["slow_steps"] = {
            "over_100ms": sum(d > 100 for d, _ in steps),
            "with_prefill": sum(d > 100 and pre > 0 for d, pre in steps),
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/summary.json"))
    a = ap.parse_args()
    runs = {}
    for run in sorted((a.results / "bench").glob("q35_9b_*")):
        if m := re.fullmatch(r"q35_9b_(\w+?)_(vllm|sglang)", run.name):
            runs[f"{m[1]}/{m[2]}"] = run_summary(run)
    summary = {
        "usd_per_gpu_hour": USD_PER_GPU_HOUR,
        "calibration": json.loads((a.results / "calib/stats.json").read_text()),
        "runs": runs,
        "traces": {
            k: trace for k, name in TRACES.items() if (trace := trace_summary(a.results / "bench" / name)) is not None
        },
    }
    a.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
