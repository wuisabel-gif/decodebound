# Morpheus

**A reproducible harness for characterizing LLM *serving* performance.**

Decompose inference into compute-bound prefill and memory-bound decode, map the
throughput / latency Pareto frontier, and treat latency as a *distribution* — not a single
averaged number that hides the tail.

`vLLM` · `continuous batching` · `Python` · roofline-oriented profiling

---

## What this is (and isn't)

This **is** a measurement instrument: it runs a real serving stack, sweeps the knobs that
matter, and characterizes the result with enough statistical care that the numbers survive
scrutiny. It does **not** train models, serve production traffic, or reinvent the load
generator — it builds an analysis layer on top of vLLM's serving benchmark.

Three questions it answers:

1. Where does inference time actually go — prefill or decode?
2. What's the real throughput/latency tradeoff, and where's the honest operating point?
3. How badly does the tail degrade under load, and *why*?

The whole point, in one line: **past the knee of the concurrency sweep, throughput stops
improving while p99 inter-token latency keeps climbing. The mean hides that regression; the
distribution doesn't.** This harness is built to make that visible and defensible.

---

## 1 · Prefill vs. decode: two different machines

Inference has two phases with opposite hardware profiles. Prefill processes the whole prompt
at once and **saturates the GPU** (compute-bound, high arithmetic intensity). Decode emits one
token at a time and **leaves the GPU mostly idle**, gated by memory bandwidth. Optimizing
inference is mostly optimizing decode.

| Phase   | Captured as            | Roofline regime          |
|---------|------------------------|--------------------------|
| Prefill | TTFT (time to first token) | compute-bound        |
| Decode  | ITL (per-token latency)    | memory-bandwidth-bound |

The harness keeps these two strictly separate — they are never merged into one "latency" —
and records GPU utilization alongside, so the memory-bound claim for decode is *measured*, not
asserted.

---

## 2 · The throughput / latency Pareto frontier

Sweeping concurrency traces the frontier every deployment negotiates. Throughput saturates
long before latency does — so the **knee** is the honest operating point. Everything to the
right buys throughput with tail latency. The harness detects the knee automatically and reports
it as the recommended operating concurrency.

---

## 3 · Where the mean lies: p50 vs. p99 under load

The signature result. As concurrency rises, the **median barely moves** — so an averaged
benchmark reports "fine." Meanwhile **p99 detaches and climbs**. Reporting a single number
would erase the tail-latency regression entirely.

---

## Methodology — why one number lies

The credibility *is* the product. This harness refuses to report a single averaged latency.

- **Full distribution, always.** Every latency figure carries p50 / p95 / p99.
- **Automatic warmup detection.** Warmup is found via the MSER-5 rule (a marginal-standard-error
  changepoint) and discarded — never eyeballed, never a hardcoded "drop first N."
- **Latency is correlated, not i.i.d.** Consecutive request latencies are coupled through
  KV-cache occupancy and scheduler state. The harness measures the **autocorrelation** between
  successive requests and reports the integrated autocorrelation time, rather than pretending
  samples are independent.
- **Convergence window.** An Allan-variance-style check reports the number of requests at which
  measured throughput actually stabilizes — so the run length is justified, not arbitrary.
- **Utilization recorded alongside latency.** A background sampler polls `nvidia-smi`
  (~4 Hz) throughout the sweep — SM-busy % (compute pressure) and memory-controller-busy %
  (the bandwidth-bound signal for decode) — timestamped on the same clock as every
  request, written to `results/raw/gpu_util.parquet`, and summarized per sweep point in
  `run_meta.json`. The roofline claims are measured, not asserted.
- **Coordinated omission, addressed.** A closed-loop worker pool slows its arrivals
  whenever the server slows down, silently absorbing queueing delay and understating the
  tail. `morpheus sweep --arrival-rate 0.5,1,2,4` runs an **open-loop Poisson sweep**
  instead: arrival times are pre-drawn (seeded, reproducible) and never conditioned on
  completions, so queueing delay lands in TTFT/ITL where it belongs. Closed loop answers
  "throughput at a pinned batch size"; open loop is the mode to trust for tail-latency
  claims. Analyze with `morpheus analyze --mode open`.

This methodology comes from a background in stochastic-process characterization (state
estimation / Allan-variance analysis), applied here to inference serving.

---

## How it compares

Mature harnesses already exist, and they beat Morpheus on **breadth** — backends, datasets,
traffic models, cluster orchestration. [guidellm](https://github.com/vllm-project/guidellm) and
NVIDIA [aiperf](https://github.com/ai-dynamo/aiperf) are the reference tools to reach for if you
need many backends or SLO-driven sweeps out of the box. Morpheus doesn't compete on that
axis. It leans into a narrower one: **treating each sweep point as a stochastic process and
reporting whether the measurement itself is trustworthy** — warmup removed by a changepoint rule,
sample independence checked rather than assumed, run length justified by a convergence criterion.

| Capability | guidellm | aiperf | Morpheus |
|---|:---:|:---:|:---:|
| Full TTFT/ITL/E2E distributions | ✅ | ✅ | ✅ |
| Open-loop (Poisson) arrivals | ✅ | ✅ | ✅ |
| Prefill/decode split as a first-class axis | partial | partial | ✅ |
| Automatic warmup detection (MSER-5) | ❌ | ❌ | ✅ |
| Autocorrelation / effective sample size | ❌ | ❌ | ✅ |
| Allan-variance convergence window | ❌ | ❌ | ✅ |
| Certifies whether the numbers are trustworthy | ❌ | ❌ | ✅ `certify` |
| Many backends / datasets / cluster scale | ✅ | ✅ | ❌ |

The claim is not "better benchmark." It's "the same numbers, with the statistical hygiene that
tells you when to believe them" — and that stats layer is what these tools don't have.

---

## Certify — should you believe your benchmark?

Every harness will print `p99 = 212ms`. None of them will tell you whether that number is a
statistic or an anecdote. `morpheus certify` turns the stats layer into **verdicts**:

```
$ morpheus certify --raw results/raw
── ITL  @ concurrency=8 ──────────────────────────────────
n=32640 → warmup 160 dropped → steady 32480
tau_int=4.2 → effective sample size 7733
mean 24.3 ms ±1.1% (95% CI)   OK
p99 61.0 ms, tail support 77.3 eff. samples   OK
drift: halves differ 1.2% (z=0.9)   OK
VERDICT: TRUSTED
```

Three failure modes it catches, each of which leaves the reported number looking perfectly normal:

- **UNDERPOWERED** — autocorrelation shrinks n to an *effective* sample size of n/τ_int; a p99
  backed by 3 effective tail samples is a dice roll. The card reports how many requests a
  trustworthy run would need.
- **NONSTATIONARY** — if the run's halves disagree beyond their (ESS-corrected) error bars, or
  MSER's truncation search hits its cap, conditions changed mid-run: thermal throttling, a noisy
  neighbor on a shared cloud GPU. The run blends two regimes; no summary of it is meaningful.
- **Warmup-dominated** — the transient ate so much of the run that "steady state" is guesswork.

It exits non-zero unless every series is TRUSTED, so it works as a **CI gate** for performance
regressions. And it isn't limited to Morpheus's own runs — feed it any harness's raw
per-request latencies and it will referee those too:

```
$ morpheus certify --file latencies.json   # a JSON array or single-column CSV, in time order
```

That's the position in the ecosystem: guidellm and aiperf are stopwatches — this is the
calibration lab. Not a competitor for the load-generation crown; the referee that any of these
tools' output can be checked against.

---

## Predict — throughput at operating points you didn't run

A sweep measures a handful of concurrencies. `morpheus predict` fits the
**Universal Scalability Law** (throughput = λN / (1 + α(N−1) + βN(N−1)) — α is contention,
β is coherency) and predicts the points you *didn't* run, with a bootstrap confidence band
and a hard label for interpolation vs. extrapolation:

```
$ morpheus predict --raw results/raw --at 3,6,12,24,32
 concurrency  throughput_tok_s  band_lo  band_hi       region
           3             170.9    170.9    174.2 INTERPOLATED
          12             539.5    538.0    539.5 INTERPOLATED
          24             792.4    792.1    805.1 EXTRAPOLATED
          32             869.8    869.1    904.5 EXTRAPOLATED
Predicted throughput-optimal concurrency (knee): 47.8
```

Two things make it honest rather than a blind curve-fit:

- **The band is a bootstrap over the raw per-request data**, not a fragile 5-point
  covariance — resample requests within each point, re-fit, take the percentile band. It
  widens where fewer (effective) samples backed a point, and blows up past the measured range.
- **Extrapolation is labelled, never hidden.** Predictions outside the measured concurrency
  range are tagged `EXTRAPOLATED` — the wide band alone is easy to gloss over, so it's named.
  (The knee above, ~48, is itself an extrapolation from data that only reached 16 — trust it
  accordingly.)

This is measured-data-backed prediction of un-run operating points — something neither the
load-generators nor the throughput-prediction literature ships in a runnable form.

---

## Results

First committed run: **Qwen2.5-1.5B-Instruct, fp16, vLLM 0.24.0, 1× Tesla T4** (a free Kaggle
GPU), 128 requests × 5 concurrency points, 512 in / 256 out. Full provenance in
[results/raw/run_meta.json](results/raw/run_meta.json); figures in `results/figures/`.

| conc | tok/s | ITL p50 | ITL p99 | TTFT p50 | TTFT p99 |
|---:|---:|---:|---:|---:|---:|
| 1  | 48  | 20.4 | 22.1 | 110 | 115 |
| 2  | 120 | 16.6 | 17.3 | 52  | 62  |
| 4  | 224 | 17.6 | 18.6 | 49  | 55  |
| 8  | 398 | 19.9 | 21.6 | 70  | 85  |
| 16 | 648 | 24.3 | 28.2 | 88  | 129 |

Knee: **concurrency 8**. Utilization was sampled throughout (~50% SM-busy, 37–48%
memory-controller-busy — a 1.5B model doesn't saturate even a T4; the decode-bound story needs
the 7B+ run).

**And `certify` flagged its own run** — which is the point:

- Every **ITL** series at c2–c16 (≈32k token events each): **TRUSTED**. Measured
  τ_int = 13–30, so 32k events collapse to an effective sample size of ~1,100–2,500 —
  the "requests are correlated" claim is now a measurement, not an argument.
- Every **TTFT p99**: **UNDERPOWERED**. 128 requests per point leaves ~1 effective tail
  sample; certify says ~2,000 are needed. The TTFT p99 column above is printed *and*
  flagged as not yet trustworthy.

A benchmark that publishes its numbers alongside the verdict that some of them don't deserve
trust yet — that's the methodology working as designed. Reproduce with `./reproduce.sh`; see
[docs/cloud-run.md](docs/cloud-run.md) for the zero-to-result walkthrough.

---

## Reproduce

```bash
pip install -e .

# run the sweep and regenerate every result and figure
./reproduce.sh --model Qwen/Qwen2.5-7B-Instruct --sweep concurrency
```

Each run writes per-request data to `results/raw/` plus a `run_meta.json`. A result without
its `run_meta.json` is not a result.

> Requires an NVIDIA GPU. On a Mac / no GPU, rent a cloud instance — only the backend changes.
> See [docs/cloud-run.md](docs/cloud-run.md).

---

## Repo layout

```
Morpheus/
├── harness/             server launch · workloads · concurrency sweep
├── analysis/            stats (percentiles, ACF, warmup, convergence) · certify · decompose · plots
├── results/             raw per-request data + figures (populated by a real run)
└── report.md            longer write-up
```

---

## Scope and honest limitations

Single-node, single-model characterization on one GPU. It does not cover multi-node /
disaggregated serving or kernel-level optimization — those are the natural next steps. Numbers
are specific to the configuration in `run_meta.json` and don't generalize across hardware.

---

<sub>Built as a study in inference performance methodology. Measurement first; the number is only as good as how it was taken.</sub>
