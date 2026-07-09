"""Launch and health-check the vLLM OpenAI-compatible server.

Hard rule from AGENTS.md: **never silently fall back to CPU.** If no CUDA GPU is
visible, this module raises :class:`NoGPUError` with instructions to rent a cloud
GPU. Detecting hardware is also how ``run_meta.json`` earns its provenance fields.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import httpx


class NoGPUError(RuntimeError):
    """Raised when no CUDA GPU is available. The run must halt, not degrade."""


@dataclass(frozen=True)
class GpuInfo:
    name: str
    memory_total_mib: int
    driver_version: str
    cuda_version: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_gpu() -> GpuInfo | None:
    """Return GPU/driver info via ``nvidia-smi``, or ``None`` if unavailable.

    Used both as a gate (no GPU -> halt) and as a provenance source for run_meta.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run(
            [
                smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    first = out.stdout.strip().splitlines()
    if not first:
        return None
    name, mem, driver = (field.strip() for field in first[0].split(","))

    cuda_version: str | None = None
    try:
        ver = subprocess.run([smi], capture_output=True, text=True, timeout=15, check=True)
        for token in ver.stdout.split():
            if token.startswith("12.") or token.startswith("11."):
                cuda_version = token
                break
    except (subprocess.SubprocessError, OSError):
        pass

    return GpuInfo(
        name=name,
        memory_total_mib=int(float(mem)),
        driver_version=driver,
        cuda_version=cuda_version,
    )


def require_gpu() -> GpuInfo:
    """Return GPU info or halt with an actionable message (no CPU fallback)."""
    info = detect_gpu()
    if info is None:
        raise NoGPUError(
            "No CUDA GPU detected (nvidia-smi unavailable).\n"
            "DecodeBound measures GPU serving and refuses to fall back to CPU — the\n"
            "numbers would be meaningless. Rent a cloud GPU (a 4090/A100 hour is cheap);\n"
            "only the serving backend changes, the rest of the repo is identical.\n"
            "See AGENTS.md > 'Hardware target' for the scaling rules."
        )
    return info


@dataclass
class ServerConfig:
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "auto"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.90
    seed: int = 0
    extra_args: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def build_launch_command(cfg: ServerConfig) -> list[str]:
    """Construct the vLLM OpenAI-server command.

    NOTE: AGENTS.md says confirm the entry point against the *installed* vLLM
    version rather than assuming flags. This is the modern entry point
    (``python -m vllm.entrypoints.openai.api_server``); verify with
    ``python -m vllm.entrypoints.openai.api_server --help`` on the target before a
    long sweep, and adjust flags here if the installed version differs.
    """
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg.model,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--dtype",
        cfg.dtype,
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--seed",
        str(cfg.seed),
        # NOTE: no --disable-log-requests — removed in newer vLLM (request logging
        # is now opt-in via --enable-log-requests, so the default is already quiet).
    ]
    if cfg.max_model_len is not None:
        cmd += ["--max-model-len", str(cfg.max_model_len)]
    cmd += list(cfg.extra_args)
    return cmd


def wait_until_healthy(
    base_url: str,
    timeout_s: float = 600.0,
    poll_s: float = 2.0,
    proc: subprocess.Popen | None = None,
) -> None:
    """Block until the server answers /health, or raise on timeout.

    vLLM cold start (weights load + CUDA-graph capture) can take minutes, hence the
    generous default. ``poll_s`` controls how often we probe. If ``proc`` is given
    and exits before the server turns healthy (bad flag, OOM, unsupported GPU),
    fail immediately instead of polling a corpse for the full timeout.
    """
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server process exited with code {proc.returncode} before "
                "becoming healthy — check its log above for the actual error."
            )
        try:
            r = httpx.get(f"{base_url}/health", timeout=5.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError as e:  # not up yet
            last_err = e
        time.sleep(poll_s)
    raise TimeoutError(f"vLLM server at {base_url} not healthy after {timeout_s}s ({last_err})")


def fetch_served_model(base_url: str) -> str | None:
    """Return the model id the server reports at /v1/models (provenance check)."""
    try:
        r = httpx.get(f"{base_url}/v1/models", timeout=10.0)
        r.raise_for_status()
        data = r.json().get("data", [])
        return data[0]["id"] if data else None
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None


@contextmanager
def serve(cfg: ServerConfig, *, launch: bool = True, startup_timeout_s: float = 600.0):
    """Context manager that brings up vLLM and tears it down.

    With ``launch=False`` it assumes a server is already running at ``cfg.base_url``
    (useful when the server lives on a separate box) and only health-checks it.
    Always requires a GPU first — the gate, not an afterthought.
    """
    require_gpu()
    proc: subprocess.Popen | None = None
    try:
        if launch:
            cmd = build_launch_command(cfg)
            proc = subprocess.Popen(cmd)  # inherits stdio so vLLM logs stream through
        wait_until_healthy(cfg.base_url, timeout_s=startup_timeout_s, proc=proc)
        yield cfg
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
