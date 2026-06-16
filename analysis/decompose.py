"""Derive aggregates from raw per-request data.

Everything here is downstream of ``results/raw/`` and never hand-edited. The two
phases are kept apart: TTFT is prefill (compute-bound), the ITL stream is decode
(memory-bound). Warmup is removed via the MSER-5 detector before any aggregate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import stats


def load_raw(raw_dir: Path | str) -> pd.DataFrame:
    """Load and concatenate all ``raw_c*.parquet`` files, parsing the ITL lists."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("raw_c*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw_c*.parquet under {raw_dir}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["itl_ms"] = df["itl_ms"].apply(lambda s: json.loads(s) if isinstance(s, str) else list(s))
    return df


def _ordered_steady(group: pd.DataFrame) -> pd.DataFrame:
    """Sort a concurrency group by start time and drop MSER-detected warmup."""
    g = group.sort_values("t_start")
    # Use per-request mean ITL as the steady-state signal (decode is the phase under load).
    signal = g["itl_ms"].apply(lambda x: float(np.mean(x)) if len(x) else np.nan).to_numpy()
    finite = signal[np.isfinite(signal)]
    if finite.size >= 10:
        cutoff = stats.detect_warmup(finite).cutoff_index
        return g.iloc[cutoff:]
    return g


@dataclass(frozen=True)
class SweepPoint:
    concurrency: int
    n_requests: int
    throughput_tok_s: float
    ttft_p50: float
    ttft_p99: float
    itl_p50: float
    itl_p95: float
    itl_p99: float
    itl_lag1_acf: float
    warmup_discarded: int


def aggregate_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """One row per concurrency: throughput + TTFT/ITL distributions (warmup removed)."""
    rows: list[dict[str, object]] = []
    for c, group in df.groupby("concurrency"):
        full = group.sort_values("t_start")
        steady = _ordered_steady(group)
        discarded = len(full) - len(steady)

        # Throughput over the steady-state wall-clock for this point.
        ends = steady["t_start"] + steady["e2e_ms"] / 1e3
        duration = float(ends.max() - steady["t_start"].min()) if len(steady) else float("nan")
        total_out = float(steady["output_tokens"].sum())
        throughput = total_out / duration if duration and duration > 0 else float("nan")

        ttft = steady["ttft_ms"].to_numpy()
        ttft = ttft[np.isfinite(ttft)]
        itl_pool = np.array([v for lst in steady["itl_ms"] for v in lst], dtype=float)

        ttft_sum = stats.percentile_summary(ttft) if ttft.size else None
        itl_sum = stats.percentile_summary(itl_pool) if itl_pool.size else None
        lag1 = stats.autocorrelation(itl_pool).lag1 if itl_pool.size > 2 else float("nan")

        rows.append(
            asdict(
                SweepPoint(
                    concurrency=int(c),
                    n_requests=int(len(steady)),
                    throughput_tok_s=throughput,
                    ttft_p50=ttft_sum.p50 if ttft_sum else float("nan"),
                    ttft_p99=ttft_sum.p99 if ttft_sum else float("nan"),
                    itl_p50=itl_sum.p50 if itl_sum else float("nan"),
                    itl_p95=itl_sum.p95 if itl_sum else float("nan"),
                    itl_p99=itl_sum.p99 if itl_sum else float("nan"),
                    itl_lag1_acf=lag1,
                    warmup_discarded=discarded,
                )
            )
        )
    return pd.DataFrame(rows).sort_values("concurrency").reset_index(drop=True)


@dataclass(frozen=True)
class PrefillDecodeSplit:
    prefill_ttft: dict[str, float | int]
    decode_itl: dict[str, float | int]
    decode_tokens_per_s: float  # 1000 / median ITL

    def as_dict(self) -> dict[str, object]:
        return {
            "prefill_ttft": self.prefill_ttft,
            "decode_itl": self.decode_itl,
            "decode_tokens_per_s": self.decode_tokens_per_s,
        }


def prefill_decode_split(df: pd.DataFrame, concurrency: int = 1) -> PrefillDecodeSplit:
    """Single-stream prefill vs decode summary at the given concurrency (default 1).

    The two-machines claim: TTFT (one prefill of the whole prompt) vs ITL (the
    per-token decode stream). Reported as distributions, not means.
    """
    g = df[df["concurrency"] == concurrency]
    if g.empty:
        raise ValueError(f"no rows at concurrency={concurrency}")
    g = _ordered_steady(g)
    ttft = g["ttft_ms"].to_numpy()
    ttft = ttft[np.isfinite(ttft)]
    itl = np.array([v for lst in g["itl_ms"] for v in lst], dtype=float)
    itl_sum = stats.percentile_summary(itl)
    return PrefillDecodeSplit(
        prefill_ttft=stats.percentile_summary(ttft).as_dict(),
        decode_itl=itl_sum.as_dict(),
        decode_tokens_per_s=1000.0 / itl_sum.p50 if itl_sum.p50 else float("nan"),
    )


def find_knee(agg: pd.DataFrame) -> int:
    """Knee of the throughput-vs-concurrency curve (Kneedle, concave-increasing).

    Normalize both axes to [0,1]; for a diminishing-returns curve the knee is the
    point furthest above the chord from first to last (max of y_norm - x_norm). That
    concurrency is the honest operating point — past it, throughput stalls while tail
    latency climbs.
    """
    a = agg.sort_values("concurrency")
    x = a["concurrency"].to_numpy(dtype=float)
    y = a["throughput_tok_s"].to_numpy(dtype=float)
    if len(x) < 3:
        return int(x[-1])
    xn = (x - x.min()) / (np.ptp(x) or 1.0)
    yn = (y - y.min()) / (np.ptp(y) or 1.0)
    return int(x[int(np.argmax(yn - xn))])
