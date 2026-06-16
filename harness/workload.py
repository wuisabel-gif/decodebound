"""Workload definitions: prompt/output-length profiles for the sweep.

Default profile is 512 input / 256 output tokens — long enough that prefill is a
real compute phase and decode is a real, measurable stream. Prompts are generated
deterministically (seeded) so a rerun replays the identical workload; the *actual*
prompt token count is whatever the server's tokenizer reports back in usage, which
the sweep records per request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# A small fixed word pool. We target a token budget by emitting words; one short
# English word is ~1.3 tokens, so we scale the word count to approximate the target.
# The exact token count is measured server-side, not assumed — this only needs to be
# in the right ballpark to exercise prefill realistically.
_WORD_POOL = (
    "model serving latency throughput decode prefill memory bandwidth compute "
    "kernel batch scheduler cache occupancy token tensor kernel roofline arithmetic "
    "intensity bound saturate utilization concurrency tail percentile distribution "
    "warmup convergence autocorrelation variance stochastic estimator sweep frontier"
).split()

_CHARS_PER_TOKEN = 4.0  # rough; used only to size the generated string


@dataclass(frozen=True)
class Workload:
    name: str
    prompt_len: int  # target prompt tokens
    output_len: int  # max tokens to generate (decode length)
    n_requests: int  # total requests to issue at each sweep point
    temperature: float = 0.0  # greedy -> deterministic decode length where possible
    seed: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# 512in / 256out default per AGENTS.md.
DEFAULT = Workload(name="default-512-256", prompt_len=512, output_len=256, n_requests=256)


def make_prompts(wl: Workload) -> list[str]:
    """Deterministically generate ``n_requests`` prompts near ``prompt_len`` tokens.

    Each prompt is a distinct seeded shuffle of the word pool repeated to the target
    length, so prompts differ (avoiding prefix-cache collapse) while the workload as
    a whole is reproducible.
    """
    rng = np.random.default_rng(wl.seed)
    approx_words = max(1, int(wl.prompt_len * _CHARS_PER_TOKEN / 6))  # ~6 chars/word incl. space
    prompts: list[str] = []
    for i in range(wl.n_requests):
        # Per-request stream so prompt i is stable regardless of n_requests.
        sub = np.random.default_rng([wl.seed, i])
        words = sub.choice(_WORD_POOL, size=approx_words, replace=True)
        prompts.append(" ".join(words.tolist()))
    del rng
    return prompts
