"""Statistical core of Morpheus.

Every function here is a pure transform over a 1-D sequence of per-request
measurements (latencies in ms, or per-request throughput). No GPU, no I/O — so
the whole module is unit-testable on synthetic data on any machine.

The four methodology pillars, each implemented as a named, defensible method:

* :func:`percentile_summary` — never report a single averaged latency. p50/p95/p99
  always travel together with the mean and spread.
* :func:`detect_warmup` — automatic transient truncation via the **Marginal
  Standard Error Rule** (MSER-5, White 1997). No eyeballing, no hardcoded "drop N".
* :func:`autocorrelation` — requests are correlated through KV-cache occupancy and
  scheduler state. We measure the ACF and the lag-1 coupling rather than assuming i.i.d.
* :func:`convergence_window` — an **overlapping Allan-variance** sweep reports the
  averaging window at which a throughput estimate actually stabilizes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ArrayLike = np.ndarray | list[float] | tuple[float, ...]


def _as_clean_array(values: ArrayLike) -> np.ndarray:
    """Coerce to a float64 array and drop NaN/inf so downstream math is safe."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


# --------------------------------------------------------------------------- #
# 1. Distribution summary — the "never report one number" rule.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DistributionSummary:
    n: int
    mean: float
    std: float
    min: float
    p50: float
    p95: float
    p99: float
    max: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def percentile_summary(
    values: ArrayLike, percentiles: tuple[float, ...] = (50.0, 95.0, 99.0)
) -> DistributionSummary:
    """Summarize a latency distribution.

    Returns p50/p95/p99 (by default) alongside mean and spread so no caller can
    accidentally collapse the distribution to a single number. Uses linear
    interpolation between order statistics (numpy default).
    """
    arr = _as_clean_array(values)
    if arr.size == 0:
        raise ValueError("percentile_summary received no finite values")
    p50, p95, p99 = (float(np.percentile(arr, p)) for p in percentiles)
    return DistributionSummary(
        n=int(arr.size),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        min=float(arr.min()),
        p50=p50,
        p95=p95,
        p99=p99,
        max=float(arr.max()),
    )


def tail_ratio(values: ArrayLike, hi: float = 99.0, lo: float = 50.0) -> float:
    """p99/p50 (by default) — a one-number *shape* metric for tail heaviness.

    A value near 1 means the tail tracks the median; large values are exactly the
    regression an averaged benchmark would hide.
    """
    arr = _as_clean_array(values)
    median = float(np.percentile(arr, lo))
    if median == 0:
        return float("inf")
    return float(np.percentile(arr, hi)) / median


# --------------------------------------------------------------------------- #
# 2. Warmup detection — MSER-5 (Marginal Standard Error Rule).
# --------------------------------------------------------------------------- #
def _mser_statistic(batched: np.ndarray) -> np.ndarray:
    """MSER objective g(d) over truncation points d on a (batched) series.

    g(d) = sum_{i>d} (Y_i - mean(Y_{d+1..n}))^2 / (n - d)^2

    The truncation that minimizes g(d) balances removing transient bias against
    keeping enough samples to estimate the mean precisely (White, 1997).
    """
    n = batched.size
    g = np.full(n, np.inf)
    # Only search the first half; truncating past n/2 is never justified.
    for d in range(n // 2):
        tail = batched[d:]
        m = tail.size
        if m < 2:
            break
        g[d] = float(np.sum((tail - tail.mean()) ** 2) / (m * m))
    return g


@dataclass(frozen=True)
class WarmupResult:
    cutoff_index: int  # in original-sample units; data[cutoff_index:] is steady state
    batch_size: int
    n_total: int
    n_discarded: int
    fraction_discarded: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def detect_warmup(values: ArrayLike, batch_size: int = 5) -> WarmupResult:
    """Find the steady-state truncation point via MSER-5.

    Data is grouped into non-overlapping batches of ``batch_size`` (5 by default,
    the standard MSER-5), the MSER objective is minimized over the batch means, and
    the winning batch boundary is mapped back to original-sample units.

    Returns the index at which steady state begins; callers discard ``data[:cutoff]``.
    This is the only sanctioned way to drop warmup in Morpheus — never a magic N.
    """
    arr = _as_clean_array(values)
    n = arr.size
    if n < 2 * batch_size:
        # Too short to truncate responsibly; keep everything.
        return WarmupResult(0, batch_size, n, 0, 0.0)

    n_batches = n // batch_size
    trimmed = arr[: n_batches * batch_size]
    batched = trimmed.reshape(n_batches, batch_size).mean(axis=1)

    g = _mser_statistic(batched)
    d_star = int(np.argmin(g))
    cutoff = d_star * batch_size
    return WarmupResult(
        cutoff_index=cutoff,
        batch_size=batch_size,
        n_total=n,
        n_discarded=cutoff,
        fraction_discarded=cutoff / n,
    )


# --------------------------------------------------------------------------- #
# 3. Autocorrelation — requests are correlated, not i.i.d.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AutocorrelationResult:
    lags: list[int]
    acf: list[float]
    lag1: float
    conf95: float  # white-noise 95% band; |acf| above this is significant
    integrated_autocorr_time: float  # tau_int; effective-sample inflation factor

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def autocorrelation(values: ArrayLike, max_lag: int | None = None) -> AutocorrelationResult:
    """Sample autocorrelation function of a latency series.

    Reports the ACF up to ``max_lag``, the lag-1 coupling, the ±1.96/sqrt(n)
    white-noise significance band, and the integrated autocorrelation time
    ``tau_int = 1 + 2*sum_{k>=1} rho_k`` (truncated at the first non-positive rho).

    ``tau_int`` is the factor by which correlation inflates the variance of the
    mean: the *effective* sample size is n / tau_int, not n. This is why an i.i.d.
    error bar on serving latency is optimistic.
    """
    arr = _as_clean_array(values)
    n = arr.size
    if n < 2:
        raise ValueError("autocorrelation needs at least 2 points")
    if max_lag is None:
        max_lag = min(n - 1, max(10, n // 4))
    max_lag = int(min(max_lag, n - 1))

    x = arr - arr.mean()
    denom = float(np.dot(x, x))
    if denom == 0:  # constant series
        acf = [1.0] + [0.0] * max_lag
    else:
        acf = [float(np.dot(x[: n - k], x[k:]) / denom) for k in range(max_lag + 1)]

    # tau_int: sum positive autocorrelations until the ACF first goes non-positive.
    tau = 1.0
    for k in range(1, max_lag + 1):
        if acf[k] <= 0:
            break
        tau += 2.0 * acf[k]

    return AutocorrelationResult(
        lags=list(range(max_lag + 1)),
        acf=acf,
        lag1=acf[1] if max_lag >= 1 else 0.0,
        conf95=float(1.96 / np.sqrt(n)),
        integrated_autocorr_time=float(tau),
    )


# --------------------------------------------------------------------------- #
# 4. Convergence window — overlapping Allan variance.
# --------------------------------------------------------------------------- #
def _overlapping_allan_var(y: np.ndarray, m: int) -> float:
    """Overlapping Allan variance at averaging length ``m`` samples.

    sigma^2(m) = 1 / (2 m^2 (N - 2m + 1)) * sum_i (S_{i+m} - 2 S_{i+... })
    computed efficiently from the cumulative sum of ``y``. ``y`` is a series of
    per-sample values (e.g. instantaneous throughput); bin means of length ``m``
    are formed at every offset (overlapping) for tighter confidence than the
    classic non-overlapping estimator.
    """
    n = y.size
    if m < 1 or 2 * m > n:
        return float("nan")
    cs = np.concatenate(([0.0], np.cumsum(y)))
    # Bin means of length m starting at every index i: mean_i = (cs[i+m]-cs[i]) / m
    bin_means = (cs[m:] - cs[:-m]) / m  # length n - m + 1
    # Differences of bin means separated by m (adjacent, non-overlapping clusters).
    diffs = bin_means[m:] - bin_means[:-m]  # length n - 2m + 1
    return float(np.sum(diffs**2) / (2.0 * (diffs.size)))


@dataclass(frozen=True)
class ConvergenceResult:
    taus: list[int]  # averaging windows in #samples
    allan_dev: list[float]  # Allan deviation at each tau
    convergence_window: int  # tau (in #requests) where the estimate stabilizes
    min_allan_dev: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def convergence_window(
    values: ArrayLike, taus: ArrayLike | None = None
) -> ConvergenceResult:
    """Allan-variance sweep → the request count at which throughput stabilizes.

    Pass a per-request series (typically instantaneous throughput = 1/latency, or
    tokens/s per request). The Allan deviation is computed over a log-spaced set of
    averaging windows ``tau`` (in number of requests). For a white-noise-dominated
    process the deviation falls as tau^-1/2; the **convergence window** is the tau
    at which it reaches its minimum — averaging longer stops helping (or drift takes
    over). That tau is the justified run length: shorter runs are noisy, longer ones
    don't tighten the estimate.
    """
    arr = _as_clean_array(values)
    n = arr.size
    if n < 4:
        raise ValueError("convergence_window needs at least 4 points")

    if taus is None:
        max_m = n // 2
        # log-spaced unique integer windows from 1 .. n/2
        candidate = np.unique(np.floor(np.logspace(0, np.log10(max_m), num=30)).astype(int))
        taus_arr = candidate[candidate >= 1]
    else:
        taus_arr = np.asarray(taus, dtype=int).ravel()

    devs: list[float] = []
    used: list[int] = []
    for m in taus_arr:
        var = _overlapping_allan_var(arr, int(m))
        if np.isfinite(var) and var >= 0:
            used.append(int(m))
            devs.append(float(np.sqrt(var)))

    if not devs:
        raise ValueError("no valid Allan-variance points; series too short")

    dev_arr = np.asarray(devs)
    conv = int(used[int(np.argmin(dev_arr))])
    return ConvergenceResult(
        taus=used,
        allan_dev=devs,
        convergence_window=conv,
        min_allan_dev=float(dev_arr.min()),
    )


# --------------------------------------------------------------------------- #
# Convenience: one call that applies all four pillars to a latency series.
# --------------------------------------------------------------------------- #
def characterize(
    values: ArrayLike, *, drop_warmup: bool = True, throughput_series: ArrayLike | None = None
) -> dict[str, object]:
    """Run the full methodology on one latency series and return a JSON-able dict.

    Pipeline: detect warmup → truncate → summarize distribution → ACF → Allan
    convergence (on ``throughput_series`` if given, else on 1/latency).
    """
    arr = _as_clean_array(values)
    warmup = detect_warmup(arr)
    steady = arr[warmup.cutoff_index :] if drop_warmup else arr

    summary = percentile_summary(steady)
    acf = autocorrelation(steady)

    thr = (
        _as_clean_array(throughput_series)
        if throughput_series is not None
        else np.where(steady > 0, 1.0 / steady, np.nan)
    )
    thr = thr[np.isfinite(thr)]
    conv = convergence_window(thr) if thr.size >= 4 else None

    return {
        "warmup": warmup.as_dict(),
        "summary": summary.as_dict(),
        "tail_ratio_p99_p50": tail_ratio(steady),
        "autocorrelation": acf.as_dict(),
        "convergence": conv.as_dict() if conv is not None else None,
    }
