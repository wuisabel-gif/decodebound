"""Tests for certification — every verdict path, on synthetic series, GPU-free."""

from __future__ import annotations

import json

import numpy as np
import pytest

from analysis import certify


def _white_noise(n: int, mean: float = 100.0, std: float = 5.0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, std, size=n)


def test_trusted_on_long_clean_series():
    c = certify.certify_series(_white_noise(20_000), label="clean")
    assert c.verdict == "TRUSTED"
    assert c.tail_support >= certify.MIN_TAIL_SUPPORT
    assert c.mean_ci_rel <= 0.05
    assert c.ess == pytest.approx(c.n_steady, rel=0.3)  # near-i.i.d. -> tau ~ 1


def test_underpowered_on_short_series():
    c = certify.certify_series(_white_noise(300), label="short")
    assert c.verdict == "UNDERPOWERED"
    # 300 samples -> ~3 effective tail samples at p99; needs ~1000.
    assert c.tail_support < certify.MIN_TAIL_SUPPORT
    assert c.needed_n >= 1000


def test_underpowered_on_tiny_series():
    c = certify.certify_series(_white_noise(10), label="tiny")
    assert c.verdict == "UNDERPOWERED"
    assert any("steady samples" in n for n in c.notes)


def test_nonstationary_on_drift():
    # Mean shifts 100 -> 160 across the run: thermal throttle / noisy neighbor.
    rng = np.random.default_rng(1)
    drift = np.linspace(100.0, 160.0, 5000) + rng.normal(0, 5, 5000)
    c = certify.certify_series(drift, label="drift")
    assert c.verdict == "NONSTATIONARY"
    assert c.drift_rel > certify.DRIFT_REL


def test_autocorrelation_shrinks_ess():
    # AR(1) with phi=0.7: tau_int ~ (1+phi)/(1-phi) ~ 5.7 -> ESS well below n.
    rng = np.random.default_rng(2)
    n, phi = 20_000, 0.7
    x = np.empty(n)
    x[0] = 0.0
    eps = rng.normal(0, 1, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    c = certify.certify_series(x * 5 + 100, label="ar1")
    assert c.tau_int > 3.0
    assert c.ess < 0.5 * c.n_steady


def test_render_contains_verdict():
    c = certify.certify_series(_white_noise(300), label="short")
    card = certify.render(c)
    assert "VERDICT: UNDERPOWERED" in card
    assert "tau_int" in card


def test_load_series_json_and_csv(tmp_path):
    values = list(_white_noise(50))
    j = tmp_path / "lat.json"
    j.write_text(json.dumps(values))
    assert certify.load_series(j) == pytest.approx(values)

    csv = tmp_path / "lat.csv"
    csv.write_text("\n".join(f"{v:.6f}" for v in values))
    assert certify.load_series(csv) == pytest.approx(values, abs=1e-5)


def test_certify_raw_closed_loop(tmp_path):
    # Reuse the synthetic closed-loop layout: raw_c*.parquet with itl_ms lists.
    import pandas as pd

    rng = np.random.default_rng(3)
    for conc in (1, 8):
        rows = []
        for i in range(60):
            itl = (20.0 + rng.gamma(2.0, 1.0, size=63)).tolist()
            ttft = 60.0 + rng.normal(0, 3)
            rows.append(
                {
                    "request_id": i, "concurrency": conc, "prompt_len_target": 512,
                    "prompt_tokens": 512, "output_tokens": 64, "ttft_ms": ttft,
                    "itl_ms": json.dumps(itl), "e2e_ms": ttft + sum(itl),
                    "t_start": float(i), "error": None,
                }
            )
        pd.DataFrame(rows).to_parquet(tmp_path / f"raw_c{conc}.parquet", index=False)

    certs = certify.certify_raw(tmp_path)
    labels = [c.label for c in certs]
    assert len(certs) == 4  # TTFT + ITL for each of 2 points
    assert any("TTFT @ concurrency=1" in label for label in labels)
    assert any("ITL" in label and "concurrency=8" in label for label in labels)
    assert all(c.verdict in {"TRUSTED", "UNDERPOWERED", "NONSTATIONARY"} for c in certs)
