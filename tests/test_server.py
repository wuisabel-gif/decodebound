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
