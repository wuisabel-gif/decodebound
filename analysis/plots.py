"""Generate the three committed figures from derived aggregates.

matplotlib only, no seaborn gimmicks. Every axis is labeled with units, and the
latency figures show p50 and p99 together — the whole point is that one number
lies. Figures are written to ``results/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; we never need an interactive window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import decompose  # noqa: E402


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_prefill_decode(split: decompose.PrefillDecodeSplit, out: Path | str) -> Path:
    """Prefill (TTFT) vs decode (median ITL): two bars, the two-machines picture."""
    out = _ensure(Path(out))
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Prefill\n(TTFT)", "Decode\n(median ITL)"]
    values = [split.prefill_ttft["p50"], split.decode_itl["p50"]]
    ax.bar(labels, values, color=["#b3331f", "#1f5fb3"])
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f} ms", ha="center", va="bottom")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Prefill vs. decode: two different machines")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_pareto(agg: pd.DataFrame, out: Path | str, knee: int | None = None) -> Path:
    """Throughput vs p99 ITL frontier, with the knee marked."""
    out = _ensure(Path(out))
    a = agg.sort_values("concurrency")
    if knee is None:
        knee = decompose.find_knee(a)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(a["throughput_tok_s"], a["itl_p99"], "-o", color="#1f5fb3")
    for _, r in a.iterrows():
        ax.annotate(f"c={int(r['concurrency'])}",
                    (r["throughput_tok_s"], r["itl_p99"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    knee_row = a[a["concurrency"] == knee]
    if not knee_row.empty:
        ax.scatter(knee_row["throughput_tok_s"], knee_row["itl_p99"],
                   s=160, facecolors="none", edgecolors="#b3331f", linewidths=2,
                   label=f"knee (c={knee})", zorder=5)
        ax.legend()
    ax.set_xlabel("system throughput (tok/s)")
    ax.set_ylabel("p99 inter-token latency (ms)")
    ax.set_title("Throughput / latency Pareto frontier")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_tail_latency(agg: pd.DataFrame, out: Path | str) -> Path:
    """p50 vs p99 ITL against concurrency — the signature 'mean lies' figure."""
    out = _ensure(Path(out))
    a = agg.sort_values("concurrency")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(a["concurrency"], a["itl_p50"], "-o", label="p50 ITL", color="#1f5fb3")
    ax.plot(a["concurrency"], a["itl_p99"], "-s", label="p99 ITL", color="#b3331f")
    ax.fill_between(a["concurrency"], a["itl_p50"], a["itl_p99"], alpha=0.08, color="#b3331f")
    ax.set_xlabel("concurrency")
    ax.set_ylabel("inter-token latency (ms)")
    ax.set_title("Where the mean lies: p50 vs p99 under load")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_all(raw_dir: Path | str = "results/raw",
                 fig_dir: Path | str = "results/figures") -> list[Path]:
    """Load raw -> aggregate -> write all three figures. Returns the paths."""
    fig_dir = Path(fig_dir)
    df = decompose.load_raw(raw_dir)
    agg = decompose.aggregate_sweep(df)
    split = decompose.prefill_decode_split(df)
    knee = decompose.find_knee(agg)
    return [
        plot_prefill_decode(split, fig_dir / "prefill_decode.png"),
        plot_pareto(agg, fig_dir / "pareto.png", knee=knee),
        plot_tail_latency(agg, fig_dir / "tail_latency.png"),
    ]
