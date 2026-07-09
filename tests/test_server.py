"""Tests for the server launch layer — no GPU, no vLLM required."""

from __future__ import annotations

import subprocess
import time

import pytest

from harness import server


def test_launch_command_has_no_removed_flags():
    # --disable-log-requests was removed in newer vLLM and killed the server at
    # parse time (observed on Kaggle). The command must stick to stable flags.
    cmd = server.build_launch_command(server.ServerConfig(model="m", max_model_len=4096))
    assert "--disable-log-requests" not in cmd
    assert cmd[:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--max-model-len" in cmd


def test_enforce_eager_flag_passthrough():
    cfg = server.ServerConfig(model="m", extra_args=("--enforce-eager",))
    assert "--enforce-eager" in server.build_launch_command(cfg)
    assert "--enforce-eager" not in server.build_launch_command(server.ServerConfig(model="m"))


def test_abort_if_server_dead():
    from harness import sweep

    def rec(i, error):
        return sweep.RequestRecord(
            request_id=i, concurrency=4, prompt_len_target=512,
            prompt_tokens=512, output_tokens=0, ttft_ms=float("nan"), error=error,
        )

    dead = [rec(i, "conn refused") for i in range(8)]
    with pytest.raises(sweep.ServerDiedError, match="server appears dead"):
        sweep._abort_if_server_dead(dead, label="c4")

    # A point with any successful request does not abort.
    mixed = dead[:-1] + [rec(7, None)]
    sweep._abort_if_server_dead(mixed, label="c4")  # no raise


def test_wait_until_healthy_fails_fast_on_dead_proc():
    # A server process that dies at startup must raise immediately with its exit
    # code, not poll a corpse until the timeout.
    proc = subprocess.Popen(["python", "-c", "raise SystemExit(2)"])
    proc.wait()
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="exited with code 2"):
        server.wait_until_healthy(
            "http://127.0.0.1:59999", timeout_s=30.0, poll_s=0.05, proc=proc
        )
    assert time.monotonic() - t0 < 5.0  # nowhere near the 30s timeout
