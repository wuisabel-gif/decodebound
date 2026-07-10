"""Certification: turn the stats layer into verdicts.

Every harness reports numbers; none tell you whether to believe them. This module
answers three questions about a per-request latency series and issues a verdict:

1. **Underpowered?** Autocorrelation shrinks the *effective* sample size to
   n / tau_int. A p99 backed by 3 effective tail samples is an anecdote, not a
   percentile. We require :data:`MIN_TAIL_SUPPORT` effective samples beyond the
   quantile and report how many total requests would be enough.
2. **Nonstationary?** If the two halves of the steady-state series disagree beyond
   what their (ESS-corrected) standard errors allow, conditions drifted mid-run —
   thermal throttling, a noisy neighbor on a shared GPU — and the run blends two
   regimes. No summary of it is trustworthy.
3. **Warmup-dominated?** MSER-5 finds the transient; if it eats a large fraction
   of the run, the "steady state" is mostly guesswork.

Verdicts: NONSTATIONARY > UNDERPOWERED > TRUSTED (worst wins).

Works on Morpheus raw parquet *or* any other harness's export via
:func:`load_series` (a JSON list or single-column CSV of latencies in time order) —
a companion to guidellm/aiperf, not a competitor.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from analysis import stats

# Thresholds — deliberately few, each defensible.
MIN_TAIL_SUPPORT = 10.0  # effective samples beyond the quantile to trust it
MIN_STEADY_N = 20  # below this, no statistic is worth computing
DRIFT_Z = 2.58  # 99% two-sided
DRIFT_REL = 0.05  # ...and the halves must differ by >5% to matter
# Strong drift inflates tau_int (the ACF stays near 1), which balloons the standard
# error and hides the drift from its own z-test. The magnitude backstop below is
# immune to that: a >15% shift between run halves is nonstationary no matter what
# the correlation structure claims.
DRIFT_REL_HARD = 0.15
WARMUP_FRACTION_WARN = 0.30
# MSER's truncation search is capped at half the run. Landing at (or near) that cap
# means the objective kept improving all the way to the boundary — i.e. the series
# never settled. That's a nonstationarity signal in its own right (White, 1997).
MSER_CAP_FRACTION = 0.45
MAX_ACF_LAG = 500  # tau_int truncates at the first non-positive rho anyway


@dataclass(frozen=True)
class Certification:
    label: str
    n_total: int
    n_warmup: int
    n_steady: int
    tau_int: float
    ess: float
    mean: float
    p99: float
    mean_ci_rel: float  # 95% CI half-width on the mean, relative (ESS-corrected)
    tail_support: float  # effective samples beyond the quantile
    needed_n: int  # steady-state requests for a trustworthy run
    drift_z: float
    drift_rel: float
    notes: list[str]
    verdict: str  # TRUSTED | UNDERPOWERED | NONSTATIONARY

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _drift_test(steady: np.ndarray, tau: float) -> tuple[float, float]:
    """Half-split z-test on the mean, with ESS-corrected standard errors.

    Returns (z, relative shift). A plain t-test would use n and flag noise —
    dividing by tau_int is the whole point.
    """
    half = steady.size // 2
    a, b = steady[:half], steady[half:]
    ess_a, ess_b = max(a.size / tau, 2.0), max(b.size / tau, 2.0)
    se = math.sqrt(a.var(ddof=1) / ess_a + b.var(ddof=1) / ess_b)
    if se == 0:
        return 0.0, 0.0
    diff = float(b.mean() - a.mean())
    center = float(steady.mean()) or 1.0
    return diff / se, abs(diff) / abs(center)


def certify_series(
    values,
    *,
    label: str = "latency",
    rel_ci_target: float = 0.05,
    quantile: float = 0.99,
) -> Certification:
    """Certify one per-request latency series (ms, in time order)."""
    arr = stats._as_clean_array(values)
    notes: list[str] = []

    warm = stats.detect_warmup(arr)
    steady = arr[warm.cutoff_index :]
    if warm.fraction_discarded > WARMUP_FRACTION_WARN:
        notes.append(
            f"warmup ate {warm.fraction_discarded:.0%} of the run — steady state is thin"
        )

    if steady.size < MIN_STEADY_N:
        return Certification(
            label=label, n_total=arr.size, n_warmup=warm.n_discarded,
            n_steady=int(steady.size), tau_int=float("nan"), ess=float(steady.size),
            mean=float(steady.mean()) if steady.size else float("nan"),
            p99=float("nan"), mean_ci_rel=float("inf"), tail_support=0.0,
            needed_n=int(math.ceil(MIN_TAIL_SUPPORT / (1 - quantile))),
            drift_z=0.0, drift_rel=0.0,
            notes=notes + [f"only {steady.size} steady samples — nothing here is a statistic"],
            verdict="UNDERPOWERED",
        )

    summ = stats.percentile_summary(steady)
    acf = stats.autocorrelation(steady, max_lag=min(MAX_ACF_LAG, steady.size - 1))
    tau = max(acf.integrated_autocorr_time, 1.0)
    ess = steady.size / tau

    mean_ci_rel = (
        1.96 * summ.std / math.sqrt(ess) / abs(summ.mean) if summ.mean else float("inf")
    )
    tail_support = ess * (1.0 - quantile)
    needed_tail = MIN_TAIL_SUPPORT / (1.0 - quantile) * tau
    needed_mean = (
        tau * (1.96 * summ.std / (rel_ci_target * abs(summ.mean))) ** 2
        if summ.mean
        else needed_tail
    )
    needed_n = int(math.ceil(max(needed_tail, needed_mean)))

    drift_z, drift_rel = _drift_test(steady, tau)

    mser_capped = warm.fraction_discarded >= MSER_CAP_FRACTION
    if mser_capped or (abs(drift_z) > DRIFT_Z and drift_rel > DRIFT_REL) \
            or drift_rel > DRIFT_REL_HARD:
        verdict = "NONSTATIONARY"
        notes.append(
            "MSER truncation hit its search cap — the series never reached steady state"
            if mser_capped
            else f"halves differ {drift_rel:.0%} (z={drift_z:.1f}) — conditions changed mid-run"
        )
    elif tail_support < MIN_TAIL_SUPPORT or mean_ci_rel > rel_ci_target:
        verdict = "UNDERPOWERED"
    else:
        verdict = "TRUSTED"

    return Certification(
        label=label, n_total=arr.size, n_warmup=warm.n_discarded,
        n_steady=int(steady.size), tau_int=float(tau), ess=float(ess),
        mean=summ.mean, p99=summ.p99, mean_ci_rel=float(mean_ci_rel),
        tail_support=float(tail_support), needed_n=needed_n,
        drift_z=float(drift_z), drift_rel=float(drift_rel),
        notes=notes, verdict=verdict,
    )


def render(c: Certification) -> str:
    """One human-readable verdict card."""
    ok = lambda cond: "OK " if cond else "LOW"  # noqa: E731 - two-token formatter
    lines = [
        f"── {c.label} " + "─" * max(1, 58 - len(c.label)),
        f"n={c.n_total} → warmup {c.n_warmup} dropped → steady {c.n_steady}",
        f"tau_int={c.tau_int:.1f} → effective sample size {c.ess:.0f}",
        f"mean {c.mean:.1f} ms ±{c.mean_ci_rel:.1%} (95% CI)"
        f"   {ok(c.mean_ci_rel <= 0.05)}",
        f"p99 {c.p99:.1f} ms, tail support {c.tail_support:.1f} eff. samples"
        f"   {ok(c.tail_support >= MIN_TAIL_SUPPORT)}"
        + (f"  (need ~{c.needed_n} steady requests)" if c.tail_support < MIN_TAIL_SUPPORT else ""),
        f"drift: halves differ {c.drift_rel:.1%} (z={c.drift_z:.1f})"
        f"   {ok(c.verdict != 'NONSTATIONARY')}",
    ]
    lines += [f"note: {n}" for n in c.notes]
    lines.append(f"VERDICT: {c.verdict}")
    return "\n".join(lines)


def load_series(path: Path | str) -> np.ndarray:
    """Load a latency series from any harness's export.

    ``.json`` → a JSON array of numbers; anything else → single-column text/CSV.
    Values are per-request latencies in ms, in time order (order matters — the
    autocorrelation and drift checks are meaningless on shuffled data).
    """
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return np.asarray(data, dtype=float)
    return np.loadtxt(path, delimiter=",", dtype=float).ravel()


def certify_raw(raw_dir: Path | str, *, mode: str = "closed") -> list[Certification]:
    """Certify every sweep point in a Morpheus raw directory (TTFT + ITL each)."""
    from analysis import decompose

    if mode == "open":
        df = decompose.load_raw(raw_dir, pattern="raw_r*.parquet")
        group_col = "arrival_rate_rps"
    else:
        df = decompose.load_raw(raw_dir)
        group_col = "concurrency"

    out: list[Certification] = []
    for key, group in df.groupby(group_col):
        g = group.sort_values("t_start")
        ttft = g["ttft_ms"].to_numpy()
        itl = np.array([v for lst in g["itl_ms"] for v in lst], dtype=float)
        out.append(certify_series(ttft, label=f"TTFT @ {group_col}={key:g}"))
        out.append(certify_series(itl, label=f"ITL  @ {group_col}={key:g}"))
    return out
