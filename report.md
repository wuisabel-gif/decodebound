# DecodeBound — longer write-up

> Companion to `README.md`. Write the prose here from real numbers once a sweep has
> run; until then this is the skeleton, with every quantitative claim left as a
> `‹measured›` placeholder so nothing ships unmeasured.

## 1. Motivation

Serving an LLM is not one workload but two, glued together. **Prefill** ingests the
whole prompt in a single forward pass — high arithmetic intensity, the GPU's compute
units saturated. **Decode** then emits one token per step, each step re-reading the
model weights and the KV cache from memory — low arithmetic intensity, bandwidth-bound,
the SMs largely idle. Reporting "latency" as one averaged number hides both this split
and, worse, the tail behaviour that actually governs user experience under load.

## 2. Method

The serving stack is vLLM (OpenAI-compatible server, continuous batching). The load
generator holds a *fixed* concurrency with a worker pool and records, per request,
TTFT (prefill) and the full inter-token-latency stream (decode), kept separate.

The analysis layer is the contribution:

- **Warmup truncation — MSER-5.** The Marginal Standard Error Rule (White, 1997)
  selects the steady-state truncation point that minimizes the marginal standard error
  of the retained mean. No eyeballing, no hardcoded "drop first N". See
  `analysis/stats.py:detect_warmup`.
- **Distribution, always.** p50/p95/p99 reported together; the p99/p50 tail ratio is a
  one-number *shape* metric. `percentile_summary`, `tail_ratio`.
- **Correlation, not i.i.d.** Consecutive request latencies are coupled through KV-cache
  occupancy and scheduler state. We report the ACF, the lag-1 coupling, and the
  integrated autocorrelation time τ_int — the factor by which correlation shrinks the
  effective sample size below n. `autocorrelation`.
- **Convergence window — Allan variance.** An overlapping Allan-deviation sweep over
  averaging windows reports the request count at which the throughput estimate
  stabilizes, justifying the run length. `convergence_window`.

Every run writes `run_meta.json` (model, dtype, vLLM version, GPU, driver, workload,
seed). Aggregates and figures are derived from `results/raw/`, never hand-edited.

## 3. Results

> Replace each `‹measured›` with a value from `decodebound analyze` and the committed
> figures. Do not paste the README's illustrative placeholders here.

### 3.1 Prefill vs. decode (single stream)

- TTFT (512-tok prompt): `‹measured›` ms
- Decode ITL (per token): `‹measured›` ms → `‹measured›` tok/s single-stream
- SM utilization, prefill vs decode: `‹measured›` % vs `‹measured›` %

![prefill vs decode](results/figures/prefill_decode.png)

### 3.2 Pareto frontier and the knee

- Knee at concurrency `‹measured›`; throughput there `‹measured›` tok/s.
- Past the knee: throughput `+‹measured›` % while p99 ITL `×‹measured›`.

![pareto](results/figures/pareto.png)

### 3.3 Tail under load

- p50 ITL across the sweep: `‹measured›` → `‹measured›` ms (barely moves).
- p99 ITL across the sweep: `‹measured›` → `‹measured›` ms (detaches).
- lag-1 ITL autocorrelation at the knee: `‹measured›`; τ_int = `‹measured›`.

![tail latency](results/figures/tail_latency.png)

## 4. Limitations

Single-node, single-model, one GPU. No multi-node / disaggregated serving, no
kernel-level work. Numbers are specific to the `run_meta.json` configuration and do not
generalize across hardware.

## 5. References

- White, K.P. (1997). *An effective truncation heuristic for bias reduction in
  simulation output.* Simulation 69(6). (MSER)
- Allan, D.W. (1966). *Statistics of atomic frequency standards.* Proc. IEEE 54(2).
