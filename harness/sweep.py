"""Concurrency sweep: a streaming load generator over the OpenAI endpoint.

For every request we record TTFT (prefill latency, time to first token) and the
full inter-token-latency (ITL) stream (decode), kept *separate* — they are two
different machines. Each sweep point holds a fixed steady-state concurrency with a
worker pool, and writes one row per request to ``results/raw/`` plus a
``run_meta.json`` capturing full provenance.

The timing math lives in :func:`compute_timing`, a pure function tested without a
network so the metric definitions are pinned down independently of the transport.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from harness import server, workload


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

    with server.serve(cfg, launch=launch_server):
        served = server.fetch_served_model(cfg.base_url) or cfg.model
        for c in concurrencies:
            records = asyncio.run(
                _run_at_concurrency(
                    cfg.base_url, served, prompts, c, wl.output_len, wl.temperature, wl.prompt_len,
                )
            )
            written.append(_write_raw(records, out_dir, tag=f"c{c}"))

    meta = _run_meta(cfg, wl, gpu, concurrencies, vllm_version)
    meta["raw_files"] = [p.name for p in written]
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path
