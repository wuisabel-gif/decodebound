# DecodeBound

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
  tail. `decodebound sweep --arrival-rate 0.5,1,2,4` runs an **open-loop Poisson sweep**
  instead: arrival times are pre-drawn (seeded, reproducible) and never conditioned on
  completions, so queueing delay lands in TTFT/ITL where it belongs. Closed loop answers
  "throughput at a pinned batch size"; open loop is the mode to trust for tail-latency
  claims. Analyze with `decodebound analyze --mode open`.

This methodology comes from a background in stochastic-process characterization (state
estimation / Allan-variance analysis), applied here to inference serving.

---

## How it compares

Mature harnesses already exist, and they beat DecodeBound on **breadth** — backends, datasets,
traffic models, cluster orchestration. [guidellm](https://github.com/vllm-project/guidellm) and
NVIDIA [aiperf](https://github.com/ai-dynamo/aiperf) are the reference tools to reach for if you
need many backends or SLO-driven sweeps out of the box. DecodeBound doesn't compete on that
axis. It leans into a narrower one: **treating each sweep point as a stochastic process and
reporting whether the measurement itself is trustworthy** — warmup removed by a changepoint rule,
sample independence checked rather than assumed, run length justified by a convergence criterion.

| Capability | guidellm | aiperf | DecodeBound |
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
statistic or an anecdote. `decodebound certify` turns the stats layer into **verdicts**:

```
$ decodebound certify --raw results/raw
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
regressions. And it isn't limited to DecodeBound's own runs — feed it any harness's raw
per-request latencies and it will referee those too:

```
$ decodebound certify --file latencies.json   # a JSON array or single-column CSV, in time order
```

That's the position in the ecosystem: guidellm and aiperf are stopwatches — this is the
calibration lab. Not a competitor for the load-generation crown; the referee that any of these
tools' output can be checked against.

---

## Results

No measured run is committed yet. On a GPU, `./reproduce.sh` produces the derived sweep table
(throughput, p50/p95/p99 TTFT and ITL, the knee) and three figures —
`prefill_decode.png`, `pareto.png`, `tail_latency.png` — written to `results/` alongside a
`run_meta.json` that captures the exact model, dtype, vLLM version, GPU, and driver. Aggregates
and figures are *derived* from the raw per-request data, never hand-edited. See
[docs/cloud-run.md](docs/cloud-run.md) for a one-evening, ~\$2 walkthrough.

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
decodebound/
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
