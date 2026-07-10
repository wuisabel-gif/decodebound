# results/

This directory holds **committed, real** measurement output:

```
results/
├── raw/         # one parquet per concurrency point (raw_c<N>.parquet) + run_meta.json
└── figures/     # prefill_decode.png · pareto.png · tail_latency.png
```

It is intentionally **empty right now** — no run has been executed on a GPU yet, and
Morpheus never commits fabricated or CPU-fallback numbers. Populate it with:

```bash
./reproduce.sh --model <hf-model-id>
```

on an NVIDIA GPU box. Every file here is *derived from* `results/raw/` and its
`run_meta.json`; figures and aggregates are never hand-edited. A result without its
`run_meta.json` is not a result.
