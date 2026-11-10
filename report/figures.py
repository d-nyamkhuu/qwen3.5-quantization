"""results/summary.json -> the README charts in docs/figures/.

    python report/figures.py

The diagram_*.svg files in the same folder are hand-drawn and share these fonts and colours.
One colour per variant everywhere; vLLM is drawn solid, SGLang outlined.
The stalls chart requires all three client traces; otherwise its previous output is removed.
"""

import itertools
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.transforms as mtransforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SANS = ["IBM Plex Sans", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
MONO = ["IBM Plex Mono", "SFMono-Regular", "Menlo", "Consolas", "DejaVu Sans Mono"]
INK, MUTED, AXIS, GRID, REF = "#1b2a3a", "#5b6472", "#b6bec9", "#e4e8ee", "#8a93a3"
VARIANTS = {"bf16": ("BF16", "#a3abb8"), "fp8": ("FP8", "#7d93b8"), "w4a16": ("W4A16", "#d0675a"),
            "nvfp4": ("NVFP4", "#2b4c7e")}  # fmt: skip
ENGINES = {"vllm": "vLLM", "sglang": "SGLang"}
FP8_KV = ("fp8", "nvfp4")  # the checkpoints whose recipe stores an FP8 KV cache
VALUE = dict(family="monospace", fontsize=9, color=INK)  # value labels at bar ends
PATCH = dict(fc="white", ec="none", pad=1)  # behind a label that would otherwise sit on a line

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "font.monospace": MONO,
        "svg.fonttype": "none",  # keep text as text: smaller, selectable, sharp
        "svg.hashsalt": "model-quantization",  # stable ids, so re-runs give identical files
        "lines.linewidth": 2.2,
        "lines.markersize": 5.5,
    }
)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # Plex is usually absent


def _style(ax, grid: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=4, color=AXIS)
    ax.grid(True, axis=grid, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)


def _figure(w: float, h: float, cols: int = 1, grid: str = "y", **kw):
    fig, axes = plt.subplots(1, cols, figsize=(w, h), facecolor="white", **kw)
    for ax in axes if cols > 1 else [axes]:
        _style(ax, grid)
    return fig, axes


def _title(fig, text: str) -> None:
    fig.suptitle(text, color=INK, fontsize=13, fontweight="bold", x=0.012, ha="left")


def _subtitle(ax, text: str) -> None:
    ax.set_title(text, color=INK, fontsize=10.5, loc="left", pad=8)


def _labels(ax, xlabel: str | None, ylabel: str | None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=10.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10.5)


def _note(ax, x, y, text: str, color: str = INK, **kw) -> None:
    ax.text(x, y, text, family="monospace", fontsize=9, color=color, **kw)


def _engine_key(fig) -> None:
    """A two-entry key: solid bar = vLLM, outlined bar = SGLang."""
    for x, (label, face) in zip((0.70, 0.84), (("vLLM", INK), ("SGLang", "white"))):
        fig.patches.append(
            plt.Rectangle((x, 0.935), 0.018, 0.03, transform=fig.transFigure, facecolor=face, edgecolor=INK, lw=1.2)
        )
        fig.text(x + 0.025, 0.95, label, fontsize=9.5, color=INK, va="center")


def _bars(ax, values: dict, fmt: str) -> None:
    """Horizontal bars per variant: vLLM solid on top, SGLang outlined below."""
    for i, v in enumerate(VARIANTS):
        label, color = VARIANTS[v]
        for j, e in enumerate(ENGINES):
            y = i + (j - 0.5) * 0.38
            x = values[f"{v}/{e}"]
            solid = e == "vllm"
            ax.barh(y, x, height=0.34, color=color if solid else "white", edgecolor=color, lw=1.4)
            ax.annotate(fmt.format(x), (x, y), xytext=(4, 0), textcoords="offset points", va="center", **VALUE)
    ax.set_yticks(range(len(VARIANTS)), [VARIANTS[v][0] for v in VARIANTS])
    if not ax.yaxis_inverted():  # shared y axes invert together
        ax.invert_yaxis()


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, facecolor="white", metadata={"Date": None})
    plt.close(fig)


def speedup(s: dict, out: Path) -> None:
    """Output tokens/s relative to BF16 on the same engine, at each concurrency."""
    fig, axes = _figure(9, 4.1, cols=2, sharey=True)
    for ax, e in zip(axes, ENGINES):
        base = s["runs"][f"bf16/{e}"]["perf"]
        xs = [int(c) for c in base]
        ends = {}
        for v, (label, color) in list(VARIANTS.items())[1:]:
            p = s["runs"][f"{v}/{e}"]["perf"]
            ys = [p[c]["output_throughput"] / base[c]["output_throughput"] for c in base]
            ax.plot(xs, ys, color=color, marker="o", markerfacecolor=color if e == "vllm" else "white")
            ends[label] = ys[-1]
        prev = None  # label each line at its right end, nudged apart so close ends stay readable
        for label, y in sorted(ends.items(), key=lambda kv: kv[1]):
            y = y if prev is None else max(y, prev + 0.09)
            ax.text(xs[-1] * 1.18, y, label, va="center", fontsize=9.5, color=INK, bbox=PATCH)
            prev = y
        ax.axhline(1, color=VARIANTS["bf16"][1], lw=2, ls="--", zorder=1)
        _note(ax, 1, 1.03, "BF16", MUTED, va="bottom")
        ax.text(1.25, 2.47, "few requests: the smallest weights win", fontsize=9, color=MUTED, va="top")
        ax.text(330, 0.65, "many requests: low-precision math wins", fontsize=9, color=MUTED, ha="right")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_xlim(0.8, 700)
        ax.set_ylim(0.6, 2.5)
        _subtitle(ax, ENGINES[e])
        _labels(ax, "concurrent requests", "output tokens/s vs BF16 (×)" if e == "vllm" else None)
    _title(fig, "Speedup over BF16 depends on load (1,024 tokens in, 512 out)")
    _save(fig, out, "speedup.svg")


def capacity_cost(s: dict, out: Path) -> None:
    """KV-cache capacity and cost per 1M output tokens, side by side; rows marked when the cache is FP8."""
    fig, (left, right) = _figure(9, 3.9, cols=2, grid="x", sharey=True)
    _bars(left, {k: r["memory"]["kv_tokens"] / 1e6 for k, r in s["runs"].items()}, "{:.2f}M")
    _bars(right, {k: r["peak"]["usd_per_1m_output_tok"] * 100 for k, r in s["runs"].items()}, "{:.1f}¢")
    tr = mtransforms.blended_transform_factory(left.transAxes, left.transData)
    for i, v in enumerate(VARIANTS):
        if v in FP8_KV:
            left.text(-0.035, i + 0.3, "FP8 KV", transform=tr, ha="right", va="center", fontsize=8, color=MUTED)
    left.set_xlim(0, 4.4)
    right.set_xlim(0, 18)
    right.tick_params(axis="y", length=0)
    _subtitle(left, "KV-cache capacity, same memory setting")
    _subtitle(right, "Cost per 1M output tokens, at peak throughput")
    _labels(left, "million tokens", None)
    _labels(right, f"US cents, at ${s['usd_per_gpu_hour']:.2f} per GPU-hour", None)
    _title(fig, "Memory and money")
    _engine_key(fig)
    _save(fig, out, "capacity_cost.svg")


def accuracy(s: dict, out: Path) -> None:
    """Every suite, every run, with ±1 standard error; the band is BF16 on vLLM ± 1 SE."""
    suites = [("math", "acc", "MATH-100", 100), ("gpqa", "acc", "GPQA-Diamond-100", 100),
              ("bfcl", "acc", "BFCL non-live", 1000), ("json", "correct", "JSON exact match", 100)]  # fmt: skip
    fig, axes = _figure(9, 3.4, cols=4, grid="x")
    for ax, (suite, key, title, n) in zip(axes, suites):
        base = s["runs"]["bf16/vllm"]["evals"][suite][key] * 100
        base_se = (base * (100 - base) / n) ** 0.5
        ax.axvspan(base - base_se, base + base_se, color="#eef1f5", lw=0)
        for i, v in enumerate(VARIANTS):
            label, color = VARIANTS[v]
            for j, e in enumerate(ENGINES):
                y = i + (j - 0.5) * 0.36
                acc = s["runs"][f"{v}/{e}"]["evals"][suite][key] * 100
                se = (acc * (100 - acc) / n) ** 0.5
                ax.errorbar(acc, y, xerr=se, fmt="o", color=color, ms=6, lw=1.4, capsize=0,
                            markerfacecolor=color if e == "vllm" else "white")  # fmt: skip
        ax.set_yticks(range(len(VARIANTS)), [VARIANTS[v][0] for v in VARIANTS] if suite == "math" else [])
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        mid = round(base / 5) * 5
        ax.set_xlim(mid - 12.5, mid + 12.5)
        _subtitle(ax, title)
        _labels(ax, "% correct", None)
    _title(fig, "Accuracy vs BF16 (±1 standard error)")
    _engine_key(fig)
    _save(fig, out, "accuracy.svg")


def stalls(s: dict, out: Path) -> None:
    """NVFP4 at 128 concurrent requests: tokens one request receives over time, and p99 against each request's worst gap."""
    configs = [("vllm", "vLLM (default)", VARIANTS["nvfp4"][1]),
               ("vllm_2048", "vLLM, 2,048-token step budget", "#8597bb"),
               ("sglang", "SGLang (default)", VARIANTS["w4a16"][1])]  # fmt: skip
    fig, (left, right) = _figure(9, 3.9, cols=2, grid="both", gridspec_kw={"width_ratios": [1.15, 1]})
    end = 5.0  # seconds shown: one wave of new prompts and its aftermath
    for key, label, color in configs:  # a flat stretch is a stall
        t = [0.0] + [x / 1000 for x in itertools.accumulate(s["traces"][key]["example_request_gaps_ms"])]
        n = sum(x <= end for x in t)
        left.step(t[:n] + [end], list(range(1, n + 1)) + [n], where="post", color=color, lw=2)
        short = label.split(" (")[0].replace("-token step budget", "")
        left.text(end + 0.08, n, short, va="center", fontsize=9, color=color)
    stall = max(s["traces"]["sglang"]["example_request_gaps_ms"], default=0) / 1000
    text = f"SGLang: nothing for {stall:.1f} s"
    coral = configs[2][2]
    arrow = dict(arrowstyle="->", color=coral, lw=1)
    left.annotate(text, (stall / 2, 2), xytext=(stall / 2, 58), ha="center", fontsize=9, color=coral, arrowprops=arrow)
    left.set_xlim(0, end + 1.1)
    left.set_xticks(range(6))
    left.set_ylim(0, 170)
    _subtitle(left, "Tokens one request receives (flat = stalled)")
    _labels(left, "seconds since its first token", "tokens received")
    right.grid(False, axis="y")
    for i, (key, label, color) in enumerate(configs):
        t = s["traces"][key]
        p99, worst = t["itl_ms"]["p99"], t["worst_gap_per_request_ms"]["median"]
        right.plot([p99, worst], [i, i], color=color, lw=2, zorder=1)
        right.plot(p99, i, "|", ms=16, color=color, mew=2.5)
        right.plot(worst, i, "o", ms=8, color=color)
        _note(right, p99 / 1.25, i, f"{p99:,.0f}", MUTED, ha="right", va="center")
        _note(right, worst * 1.25, i, f"{worst:,.0f}", INK, ha="left", va="center")
    right.text(21, -0.45, "p99 of all gaps", fontsize=9, color=MUTED, ha="center")
    right.text(1350, -0.45, "worst gap of the\nmedian request", fontsize=9, color=INK, ha="center")
    right.set_xscale("log")
    right.minorticks_off()
    right.set_xlim(6, 9000)
    right.set_xticks([10, 30, 100, 300, 1000, 3000], ["10", "30", "100", "300", "1k", "3k"])
    right.set_yticks(range(3), [label for _, label, _ in configs])
    right.set_ylim(2.5, -0.9)
    right.tick_params(axis="y", length=0)
    _subtitle(right, "p99 against each request's worst gap")
    _labels(right, "ms (log scale)", None)
    _title(fig, "Where the latency tail comes from (NVFP4, 128 concurrent requests)")
    _save(fig, out, "stalls.svg")


def main(summary: Path = Path("results/summary.json"), out: Path = Path("docs/figures")) -> None:
    s = json.loads(summary.read_text())
    out.mkdir(parents=True, exist_ok=True)
    for chart in (speedup, capacity_cost, accuracy):
        chart(s, out)
    missing = {"vllm", "vllm_2048", "sglang"} - s.get("traces", {}).keys()
    if missing:
        (out / "stalls.svg").unlink(missing_ok=True)
        print(f"skipping stalls.svg: missing client traces: {', '.join(sorted(missing))}")
    else:
        stalls(s, out)
    print(f"wrote charts to {out}/")


if __name__ == "__main__":
    main()
