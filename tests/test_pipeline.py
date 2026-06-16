"""Integration tests for the derived pipeline, GPU-free.

We synthesize raw per-request parquet that *mimics* a real sweep — a warmup
transient plus a tail that worsens with concurrency — then run the real
aggregate -> decompose -> plot path. This exercises everything downstream of the
serving stack without a GPU, and never writes into the committed results/ tree
(synthetic numbers must not masquerade as measurements).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from analysis import decompose, plots
from harness import sweep, workload


# --------------------------------------------------------------------------- #
# Pure timing math
# --------------------------------------------------------------------------- #
def test_compute_timing_basic():
    # Tokens arrive at t=1.0 (TTFT vs start 0.5) then every 0.02s.
    t_start = 0.5
    token_times = [1.0, 1.02, 1.05, 1.09]
    t_end = 1.10
    timing = sweep.compute_timing(token_times, t_start, t_end)
    assert timing.ttft_ms == pytest.approx(500.0)  # (1.0 - 0.5) * 1000
    assert timing.itl_ms == pytest.approx([20.0, 30.0, 40.0])
    assert timing.e2e_ms == pytest.approx(600.0)
    assert timing.n_token_events == 4


def test_compute_timing_no_tokens():
    timing = sweep.compute_timing([], 0.0, 1.0)
    assert np.isnan(timing.ttft_ms)
    assert timing.itl_ms == []
    assert timing.e2e_ms == pytest.approx(1000.0)


def test_workload_prompts_deterministic():
    wl = workload.Workload("t", prompt_len=128, output_len=32, n_requests=8, seed=7)
    a = workload.make_prompts(wl)
    b = workload.make_prompts(wl)
    assert a == b  # seeded -> reproducible
    assert len({*a}) > 1  # prompts differ from each other


# --------------------------------------------------------------------------- #
# Synthetic raw data generator
# --------------------------------------------------------------------------- #
def _synth_raw(tmp_path, concurrencies=(1, 4, 8, 16, 32, 48), n=120, seed=1):
    """Write raw_c*.parquet that looks like a real (transient + load-coupled) sweep."""
    rng = np.random.default_rng(seed)
    for c in concurrencies:
        # ITL median rises gently with load; the p99 tail blows up super-linearly.
        base_itl = 20.0 + 0.4 * c
        tail_scale = 1.0 + (c / 16.0) ** 2  # heavy tail past the knee
        rows = []
        t = 0.0
        for i in range(n):
            warmup = 40.0 * np.exp(-i / 8.0)  # decaying transient, MSER should catch it
            out_tokens = 64
            itl = (
                base_itl
                + warmup
                + rng.gamma(shape=2.0, scale=tail_scale, size=out_tokens - 1)
            ).tolist()
            ttft = 60.0 + 1.5 * c + warmup + rng.normal(0, 3)
            e2e = (ttft + float(np.sum(itl))) / 1e3
            rows.append(
                {
                    "request_id": i,
                    "concurrency": c,
                    "prompt_len_target": 512,
                    "prompt_tokens": 512,
                    "output_tokens": out_tokens,
                    "ttft_ms": ttft,
                    "itl_ms": json.dumps(itl),
                    "e2e_ms": e2e * 1e3,
                    "t_start": t,
                    "error": None,
                }
            )
            t += e2e / max(c, 1)  # higher concurrency -> tighter request spacing
        pd.DataFrame(rows).to_parquet(tmp_path / f"raw_c{c}.parquet", index=False)


def test_aggregate_and_knee(tmp_path):
    _synth_raw(tmp_path)
    df = decompose.load_raw(tmp_path)
    agg = decompose.aggregate_sweep(df)

    # One row per concurrency, sorted.
    assert list(agg["concurrency"]) == sorted(agg["concurrency"])
    # Tail must widen with load: p99/p50 ratio grows from low to high concurrency.
    ratio_lo = agg.iloc[0]["itl_p99"] / agg.iloc[0]["itl_p50"]
    ratio_hi = agg.iloc[-1]["itl_p99"] / agg.iloc[-1]["itl_p50"]
    assert ratio_hi > ratio_lo
    # Warmup detector dropped *something* on at least one point.
    assert agg["warmup_discarded"].sum() > 0
    # Knee is an actual sampled concurrency.
    knee = decompose.find_knee(agg)
    assert knee in set(agg["concurrency"])


def test_prefill_decode_split(tmp_path):
    _synth_raw(tmp_path)
    df = decompose.load_raw(tmp_path)
    split = decompose.prefill_decode_split(df, concurrency=1)
    # Prefill (whole-prompt TTFT) >> single decode step (per-token ITL).
    assert split.prefill_ttft["p50"] > split.decode_itl["p50"]
    assert split.decode_tokens_per_s > 0


def test_generate_all_figures(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    fig = tmp_path / "figures"
    _synth_raw(raw)
    paths = plots.generate_all(raw_dir=raw, fig_dir=fig)
    assert len(paths) == 3
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
