"""Predict un-run sweep points with a confidence band.

A sweep measures a handful of concurrency points. This predicts the throughput at
the concurrencies you *didn't* run, and — the honest part — a confidence band that
widens where the prediction is extrapolating beyond the measured range.

Two ideas, borrowed and adapted:

* **Model = Universal Scalability Law** (Gunther), the physically-motivated curve for
  throughput vs. concurrency::

      throughput(N) = lambda*N / (1 + alpha*(N-1) + beta*N*(N-1))

  ``alpha`` is contention (queueing → saturation), ``beta`` is coherency (cross-request
  coordination → eventual retrograde). Three parameters, fit with ``scipy.curve_fit``.
  Unlike a blind polynomial it extrapolates with a physical basis, and the throughput-
  maximizing concurrency has a closed form: ``N* = sqrt((1-alpha)/beta)``.

* **Band = nonparametric bootstrap over the raw per-request data.** Morpheus keeps
  thousands of per-request timings behind each point, so instead of a fragile 5-point
  covariance we resample requests within each point, re-aggregate, re-fit, and take the
  percentile band of the resulting family of curves. The band correctly widens where
  fewer (effective) samples backed a point, and blows up in the extrapolation region.

The paper's one hard lesson (ML "struggles extrapolating beyond training data") becomes a
label: predictions inside the measured concurrency range are ``INTERPOLATED``; outside,
``EXTRAPOLATED`` — a wide band is easy to gloss over, so we name it too.

# ponytail: USL assumes a single saturating resource — valid for one GPU / one model,
# NOT across hardware. Fit is 3 params on ~5 points; the bootstrap band is what keeps it
# honest. Upgrade path: hierarchical fit across multiple runs if you ever sweep hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def usl(n: np.ndarray, lam: float, alpha: float, beta: float) -> np.ndarray:
    """Universal Scalability Law throughput at concurrency ``n``."""
    n = np.asarray(n, dtype=float)
    return lam * n / (1.0 + alpha * (n - 1.0) + beta * n * (n - 1.0))


@dataclass(frozen=True)
class UslFit:
    lam: float
    alpha: float
    beta: float
    c_min: float  # measured concurrency range — the interpolation boundary
    c_max: float

    def predict(self, n: np.ndarray) -> np.ndarray:
        return usl(n, self.lam, self.alpha, self.beta)

    @property
    def knee(self) -> float:
        """Throughput-maximizing concurrency, N* = sqrt((1-alpha)/beta).

        Undefined (nan) when the coherency term is negligible (beta ≲ 1e-6): the curve
        just saturates with no interior maximum, so a "knee" of tens of thousands would
        be a fit artifact, not a real operating point. Honest answer: no retrograde here.
        """
        if self.beta <= 1e-6 or self.alpha >= 1:
            return float("nan")
        return float(np.sqrt((1.0 - self.alpha) / self.beta))


def _fit_points(concurrency: np.ndarray, throughput: np.ndarray) -> tuple[float, float, float]:
    """Least-squares USL fit; returns (lam, alpha, beta). Raises on <3 points."""
    x = np.asarray(concurrency, dtype=float)
    y = np.asarray(throughput, dtype=float)
    if x.size < 3:
        raise ValueError("USL fit needs at least 3 concurrency points")
    lam0 = float(y[0] / x[0]) if x[0] else float(y.max())
    popt, _ = curve_fit(
        usl, x, y,
        p0=[lam0, 0.1, 0.01],
        bounds=([0.0, 0.0, 0.0], [np.inf, 10.0, 10.0]),
        maxfev=20000,
    )
    return float(popt[0]), float(popt[1]), float(popt[2])


def fit_usl(agg: pd.DataFrame) -> UslFit:
    """Fit the USL to an aggregated sweep (``concurrency`` + ``throughput_tok_s``)."""
    a = agg.sort_values("concurrency")
    c = a["concurrency"].to_numpy(dtype=float)
    lam, alpha, beta = _fit_points(c, a["throughput_tok_s"].to_numpy(dtype=float))
    return UslFit(lam=lam, alpha=alpha, beta=beta, c_min=float(c.min()), c_max=float(c.max()))


def _throughput_of(group: pd.DataFrame) -> float:
    """Aggregate tok/s of a raw per-request group (matches decompose._group_metrics)."""
    ends = group["t_start"] + group["e2e_ms"] / 1e3
    duration = float(ends.max() - group["t_start"].min())
    total_out = float(group["output_tokens"].sum())
    return total_out / duration if duration > 0 else float("nan")


def observed_throughput(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-concurrency throughput computed the *same way the bootstrap does* (full group).

    Used as the point-estimate fit so the point and the bootstrap band share one
    throughput definition — otherwise the band can fail to bracket the point.
    """
    rows = [{"concurrency": int(c), "throughput_tok_s": _throughput_of(g)}
            for c, g in raw.groupby("concurrency")]
    return pd.DataFrame(rows).sort_values("concurrency").reset_index(drop=True)


def fit_from_raw(raw: pd.DataFrame) -> UslFit:
    """USL fit on the bootstrap-consistent observed throughput."""
    return fit_usl(observed_throughput(raw))


def bootstrap_band(
    raw: pd.DataFrame, ns, *, n_boot: int = 1000, seed: int = 0, ci: float = 95.0
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap the prediction band at concurrencies ``ns`` from raw per-request data.

    Each iteration resamples requests (with replacement) within every concurrency point,
    recomputes that point's throughput, refits the USL, and predicts at ``ns``. Returns
    ``(lo, hi)`` percentile arrays. Fits that fail to converge are skipped, not faked.
    """
    ns = np.asarray(ns, dtype=float)
    rng = np.random.default_rng(seed)
    groups = {int(c): g.reset_index(drop=True) for c, g in raw.groupby("concurrency")}
    conc = np.array(sorted(groups), dtype=float)

    curves: list[np.ndarray] = []
    for _ in range(n_boot):
        thr = []
        for c in conc:
            g = groups[int(c)]
            idx = rng.integers(0, len(g), size=len(g))
            thr.append(_throughput_of(g.iloc[idx]))
        try:
            lam, alpha, beta = _fit_points(conc, np.asarray(thr))
        except (RuntimeError, ValueError):
            continue  # a bootstrap resample that won't fit is dropped, never faked
        curves.append(usl(ns, lam, alpha, beta))

    if not curves:
        nan = np.full(ns.shape, np.nan)
        return nan, nan
    stacked = np.vstack(curves)
    half = (100.0 - ci) / 2.0
    return np.percentile(stacked, half, axis=0), np.percentile(stacked, 100.0 - half, axis=0)


def predict_table(
    raw: pd.DataFrame, ns, *, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Predicted throughput + bootstrap band + interpolation region tag at ``ns``.

    The point fit and the bootstrap share one throughput definition (full group), so
    the band always brackets the point.
    """
    fit = fit_from_raw(raw)
    ns = np.asarray(ns, dtype=float)
    point = fit.predict(ns)
    lo, hi = bootstrap_band(raw, ns, n_boot=n_boot, seed=seed)
    region = np.where(
        (ns >= fit.c_min) & (ns <= fit.c_max), "INTERPOLATED", "EXTRAPOLATED"
    )
    return pd.DataFrame(
        {
            "concurrency": ns.astype(int) if np.all(ns == ns.astype(int)) else ns,
            "throughput_tok_s": point,
            "band_lo": lo,
            "band_hi": hi,
            "region": region,
        }
    )
