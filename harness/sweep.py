"""Concurrency sweep: a streaming load generator over the OpenAI endpoint.

For every request we record TTFT (prefill latency, time to first token) and the
full inter-token-latency (ITL) stream (decode), kept *separate* — they are two
different machines. Each sweep point holds a fixed steady-state concurrency with a
worker pool, and writes one row per request to ``results/raw/`` plus a
``run_meta.json`` capturing full provenance.

The timing math lives in :func:`compute_timing`, a pure function tested without a
network so the metric definitions are pinned down independently of the transport.

Two load modes:

* **Closed loop** (:func:`run_sweep`) — a worker pool holds a fixed concurrency.
* **Open loop** (:func:`run_open_loop_sweep`) — requests launch at pre-drawn Poisson
  arrival times regardless of completions. The closed loop slows its arrivals whenever
  the server slows down (*coordinated omission*), which understates tail latency;
  the open loop lets queueing delay land in the measurement instead.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import numpy as np

from harness import gpumon, server, workload


# --------------------------------------------------------------------------- #
# Pure timing math (network-free, unit-tested).
# --------------------------------------------------------------------------- #
@dataclass
class Timing:
    ttft_ms: float
    itl_ms: list[float]
    e2e_ms: float
    n_token_events: int


def compute_timing(
    token_times: list[float], t_start: float, t_end: float
) -> Timing:
    """Derive TTFT / ITL / e2e from absolute token-arrival timestamps (seconds).

    * ``ttft_ms`` — first token arrival minus request start (the prefill phase).
    * ``itl_ms``  — successive differences between token arrivals (the decode phase).
    * ``e2e_ms``  — request start to stream end.

    Defined here, once, so prefill and decode can never be silently merged.
    """
    if not token_times:
        return Timing(
            ttft_ms=float("nan"), itl_ms=[], e2e_ms=(t_end - t_start) * 1e3, n_token_events=0
        )
    ttft = (token_times[0] - t_start) * 1e3
    itl = [(token_times[i] - token_times[i - 1]) * 1e3 for i in range(1, len(token_times))]
    return Timing(
        ttft_ms=ttft,
        itl_ms=itl,
        e2e_ms=(t_end - t_start) * 1e3,
        n_token_events=len(token_times),
    )


@dataclass
class RequestRecord:
    request_id: int
    concurrency: int
    prompt_len_target: int
    prompt_tokens: int | None
    output_tokens: int
    ttft_ms: float
    itl_ms: list[float] = field(default_factory=list)
    e2e_ms: float = float("nan")
    t_start: float = 0.0
    error: str | None = None
    arrival_rate_rps: float | None = None  # set only in open-loop mode

    def row(self) -> dict[str, object]:
        d = asdict(self)
        # Store ITL list as JSON so it survives CSV/parquet round-trips uniformly.
        d["itl_ms"] = json.dumps(self.itl_ms)
        return d


# --------------------------------------------------------------------------- #
# Streaming a single request.
# --------------------------------------------------------------------------- #
async def _stream_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_id: int,
    concurrency: int,
    prompt_len_target: int,
) -> RequestRecord:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    token_times: list[float] = []
    prompt_tokens: int | None = None
    output_tokens = 0
    t_start = time.perf_counter()
    try:
        async with client.stream(
            "POST", f"{base_url}/v1/completions", json=payload, timeout=300.0
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                now = time.perf_counter()
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    token_times.append(now)
                    output_tokens += 1
                usage = chunk.get("usage")
                if usage:  # final chunk when include_usage is honored
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    if usage.get("completion_tokens"):
                        output_tokens = usage["completion_tokens"]
        t_end = time.perf_counter()
        timing = compute_timing(token_times, t_start, t_end)
        return RequestRecord(
            request_id=request_id,
            concurrency=concurrency,
            prompt_len_target=prompt_len_target,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=timing.ttft_ms,
            itl_ms=timing.itl_ms,
            e2e_ms=timing.e2e_ms,
            t_start=t_start,
        )
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        return RequestRecord(
            request_id=request_id,
            concurrency=concurrency,
            prompt_len_target=prompt_len_target,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=float("nan"),
            t_start=t_start,
            error=f"{type(e).__name__}: {e}",
        )


# --------------------------------------------------------------------------- #
# Holding a fixed concurrency.
# --------------------------------------------------------------------------- #
async def _run_at_concurrency(
    base_url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    temperature: float,
    prompt_len_target: int,
) -> list[RequestRecord]:
    """Issue ``len(prompts)`` requests through exactly ``concurrency`` workers.

    A worker pool pulling from a queue keeps the in-flight count pinned at the target
    (an open-loop "fire N at once" would let concurrency decay as requests finish).
    """
    queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
    for i, p in enumerate(prompts):
        queue.put_nowait((i, p))
    records: list[RequestRecord] = []

    async with httpx.AsyncClient() as client:
        async def worker() -> None:
            while True:
                try:
                    rid, prompt = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                rec = await _stream_one(
                    client, base_url, model, prompt, max_tokens, temperature,
                    rid, concurrency, prompt_len_target,
                )
                records.append(rec)

        await asyncio.gather(*[worker() for _ in range(concurrency)])
    records.sort(key=lambda r: r.t_start)
    return records


# --------------------------------------------------------------------------- #
# Open-loop (Poisson) load — the coordinated-omission-free mode.
# --------------------------------------------------------------------------- #
def poisson_offsets(n: int, rate_rps: float, seed: int = 0) -> list[float]:
    """Cumulative arrival offsets (seconds from t0) of a Poisson process at ``rate_rps``.

    Inter-arrival gaps are exponential with mean ``1/rate_rps``, drawn *ahead of
    time* and never conditioned on completions — the definition of an open loop.
    Seeded per (seed, rate) so a rerun replays the identical arrival schedule.
    Pure function; this is the tested surface of the open-loop mode.
    """
    if n < 1:
        raise ValueError("poisson_offsets needs n >= 1")
    if rate_rps <= 0:
        raise ValueError("rate_rps must be positive")
    rng = np.random.default_rng([seed, int(round(rate_rps * 1e6))])
    return np.cumsum(rng.exponential(1.0 / rate_rps, size=n)).tolist()


async def _run_at_rate(
    base_url: str,
    model: str,
    prompts: list[str],
    rate_rps: float,
    max_tokens: int,
    temperature: float,
    prompt_len_target: int,
    seed: int,
) -> list[RequestRecord]:
    """Issue ``len(prompts)`` requests at pre-drawn Poisson arrival times.

    Unlike :func:`_run_at_concurrency`, nothing here waits for a response before
    launching the next request: a slow server accumulates in-flight requests and the
    resulting queueing delay shows up in TTFT/ITL, where it belongs. This is the mode
    to trust for tail-latency claims; the closed loop is the mode to trust for
    "throughput at a pinned batch size".
    """
    offsets = poisson_offsets(len(prompts), rate_rps, seed)
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        tasks: list[asyncio.Task[RequestRecord]] = []
        for (i, prompt), offset in zip(enumerate(prompts), offsets, strict=True):
            delay = (t0 + offset) - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            # concurrency=0 marks open-loop rows; in-flight count is emergent here.
            tasks.append(
                asyncio.create_task(
                    _stream_one(
                        client, base_url, model, prompt, max_tokens, temperature,
                        i, 0, prompt_len_target,
                    )
                )
            )
        records = list(await asyncio.gather(*tasks))
    for rec in records:
        rec.arrival_rate_rps = rate_rps
    records.sort(key=lambda r: r.t_start)
    return records


class ServerDiedError(RuntimeError):
    """A sweep point saw every request fail — the server is down, so stop early."""


def _abort_if_server_dead(records: list[RequestRecord], label: object) -> None:
    """Stop the sweep if a whole point errored out.

    If the vLLM engine dies mid-sweep (e.g. the T4 FlashAttention crash), every
    request at the current point fails. Continuing would just hammer a dead server
    for hours and write all-error parquet. Fail loud instead — the points already
    written are preserved and usable.
    """
    if records and all(r.error is not None for r in records):
        raise ServerDiedError(
            f"all {len(records)} requests at {label} failed "
            f"(e.g. {records[0].error!r}) — server appears dead, aborting sweep."
        )


# --------------------------------------------------------------------------- #
# Top-level sweep + persistence.
# --------------------------------------------------------------------------- #
DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 48)


def _write_raw(records: list[RequestRecord], out_dir: Path, tag: str) -> Path:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.row() for r in records])
    path = out_dir / f"raw_{tag}.parquet"
    df.to_parquet(path, index=False)
    return path


def _run_meta(cfg: server.ServerConfig, wl: workload.Workload, gpu: server.GpuInfo,
              concurrencies: tuple[int, ...], vllm_version: str | None) -> dict[str, object]:
    return {
        "model": cfg.model,
        "dtype": cfg.dtype,
        "seed": cfg.seed,
        "vllm_version": vllm_version,
        "gpu": gpu.as_dict(),
        "workload": wl.as_dict(),
        "concurrencies": list(concurrencies),
        "max_model_len": cfg.max_model_len,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
    }


def run_sweep(
    cfg: server.ServerConfig,
    wl: workload.Workload = workload.DEFAULT,
    concurrencies: tuple[int, ...] = DEFAULT_CONCURRENCIES,
    out_dir: Path | str = "results/raw",
    *,
    launch_server: bool = True,
) -> Path:
    """Run the full concurrency sweep and persist raw data + run_meta.json.

    Returns the path to the written run_meta.json. Aggregation and plotting are
    *derived* downstream from the committed raw parquet, never hand-edited.
    """
    out_dir = Path(out_dir)
    gpu = server.require_gpu()  # halt here on no-GPU, before launching anything
    try:
        import vllm  # type: ignore

        vllm_version = getattr(vllm, "__version__", None)
    except ImportError:
        vllm_version = None

    prompts = workload.make_prompts(wl)
    written: list[Path] = []
    sampler = gpumon.GpuSampler()

    with server.serve(cfg, launch=launch_server), sampler:
        served = server.fetch_served_model(cfg.base_url) or cfg.model
        for c in concurrencies:
            sampler.set_label(f"c{c}")
            records = asyncio.run(
                _run_at_concurrency(
                    cfg.base_url, served, prompts, c, wl.output_len, wl.temperature, wl.prompt_len,
                )
            )
            written.append(_write_raw(records, out_dir, tag=f"c{c}"))
            _abort_if_server_dead(records, c)

    util_path = sampler.to_parquet(out_dir / "gpu_util.parquet")
    meta = _run_meta(cfg, wl, gpu, concurrencies, vllm_version)
    meta["mode"] = "closed-loop"
    meta["raw_files"] = [p.name for p in written]
    meta["gpu_util_file"] = util_path.name if util_path else None
    meta["gpu_util_summary"] = sampler.summary()
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def run_open_loop_sweep(
    cfg: server.ServerConfig,
    wl: workload.Workload = workload.DEFAULT,
    rates_rps: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    out_dir: Path | str = "results/raw",
    *,
    launch_server: bool = True,
) -> Path:
    """Open-loop counterpart of :func:`run_sweep`: sweep Poisson arrival rates.

    Writes one ``raw_r<rate>.parquet`` per rate plus the same provenance
    ``run_meta.json`` (with ``mode: open-loop``). Rows carry ``arrival_rate_rps``;
    the ``concurrency`` column is 0 because in-flight count is emergent, not pinned.
    """
    out_dir = Path(out_dir)
    gpu = server.require_gpu()
    try:
        import vllm  # type: ignore

        vllm_version = getattr(vllm, "__version__", None)
    except ImportError:
        vllm_version = None

    prompts = workload.make_prompts(wl)
    written: list[Path] = []
    sampler = gpumon.GpuSampler()

    with server.serve(cfg, launch=launch_server), sampler:
        served = server.fetch_served_model(cfg.base_url) or cfg.model
        for rate in rates_rps:
            sampler.set_label(f"r{rate:g}")
            records = asyncio.run(
                _run_at_rate(
                    cfg.base_url, served, prompts, rate,
                    wl.output_len, wl.temperature, wl.prompt_len, cfg.seed,
                )
            )
            written.append(_write_raw(records, out_dir, tag=f"r{rate:g}"))
            _abort_if_server_dead(records, f"rate={rate:g}")

    util_path = sampler.to_parquet(out_dir / "gpu_util.parquet")
    meta = _run_meta(cfg, wl, gpu, concurrencies=(), vllm_version=vllm_version)
    meta["mode"] = "open-loop"
    meta["arrival_rates_rps"] = list(rates_rps)
    meta["raw_files"] = [p.name for p in written]
    meta["gpu_util_file"] = util_path.name if util_path else None
    meta["gpu_util_summary"] = sampler.summary()
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path
