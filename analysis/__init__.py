"""Morpheus analysis layer — the novel work.

Pure functions over per-request latency data: distribution summaries, automatic
warmup detection, latency autocorrelation, and an Allan-variance-style convergence
window. Everything here is GPU-free and unit-tested on synthetic data, so it can be
developed and verified on any machine.
"""
