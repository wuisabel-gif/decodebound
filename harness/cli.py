"""Command-line entry point: ``morpheus <subcommand>``.

Subcommands:
  check-gpu   detect the GPU/driver, or print the halt message and exit non-zero
  sweep       launch vLLM, run the concurrency sweep, write results/raw/ + run_meta
  analyze     load raw data, print the derived sweep table + the knee
  plot        regenerate the three figures from raw data
  certify     verdict per series: should you believe this benchmark?
  predict     throughput at un-run concurrencies, with a confidence band
"""

from __future__ import annotations

import argparse
import sys


def _parse_concurrency(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def cmd_check_gpu(_: argparse.Namespace) -> int:
    from harness import server

    info = server.detect_gpu()
    if info is None:
        print(
            "No CUDA GPU detected. Morpheus will not fall back to CPU.\n"
            "Rent a cloud GPU and rerun; only the serving backend changes.",
            file=sys.stderr,
        )
        return 1
    print(f"GPU:    {info.name}")
    print(f"VRAM:   {info.memory_total_mib} MiB")
    print(f"Driver: {info.driver_version}  CUDA: {info.cuda_version}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from harness import server, sweep, workload

    cfg = server.ServerConfig(
        model=args.model,
        port=args.port,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        seed=args.seed,
        extra_args=(("--enforce-eager",) if args.enforce_eager else ()),
    )
    wl = workload.Workload(
        name=f"{args.prompt_len}in-{args.output_len}out",
        prompt_len=args.prompt_len,
        output_len=args.output_len,
        n_requests=args.n_requests,
        seed=args.seed,
    )
    open_loop = bool(args.arrival_rate)
    if open_loop:
        rates = tuple(float(x) for x in args.arrival_rate.split(",") if x.strip())
        points, kind = rates, "arrival-rate (open-loop Poisson)"
    else:
        concurrencies = _parse_concurrency(args.concurrency)
        points, kind = concurrencies, "concurrency (closed-loop)"

    n_points = len(points)
    total_requests = n_points * wl.n_requests
    print(f"Planned: {n_points} {kind} points x {wl.n_requests} requests "
          f"= {total_requests} requests against {args.model}.")
    if not args.yes:
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return 1

    if open_loop:
        meta = sweep.run_open_loop_sweep(
            cfg, wl, rates, out_dir=args.raw, launch_server=not args.no_launch
        )
    else:
        meta = sweep.run_sweep(
            cfg, wl, concurrencies, out_dir=args.raw, launch_server=not args.no_launch
        )
    print(f"Wrote {meta}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from analysis import decompose

    if args.mode == "open":
        df = decompose.load_raw(args.raw, pattern="raw_r*.parquet")
        agg = decompose.aggregate_rate_sweep(df)
        x_col, x_name = "arrival_rate_rps", "arrival rate (req/s)"
    else:
        df = decompose.load_raw(args.raw)
        agg = decompose.aggregate_sweep(df)
        x_col, x_name = "concurrency", "concurrency"
    knee = decompose.find_knee(agg, x_col=x_col)
    cols = [x_col, "n_requests", "error_rate", "throughput_tok_s",
            "itl_p50", "itl_p99", "ttft_p50", "ttft_p99", "itl_lag1_acf"]
    with __import__("pandas").option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(agg[cols].to_string(index=False))
    print(f"\nKnee (honest operating point): {x_name} = {knee:g}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from analysis import plots

    paths = plots.generate_all(raw_dir=args.raw, fig_dir=args.figures)
    for p in paths:
        print(f"Wrote {p}")
    return 0


def cmd_certify(args: argparse.Namespace) -> int:
    from analysis import certify

    if args.file:
        series = certify.load_series(args.file)
        certs = [certify.certify_series(series, label=args.file, rel_ci_target=args.rel_ci)]
    else:
        certs = certify.certify_raw(args.raw, mode=args.mode)

    for c in certs:
        print(certify.render(c))
        print()
    # Exit 0 only if every series earned TRUSTED — so CI can gate on it.
    return 0 if all(c.verdict == "TRUSTED" for c in certs) else 1


def cmd_predict(args: argparse.Namespace) -> int:
    import math

    from analysis import decompose, predict

    ns = [float(x) for x in args.at.split(",") if x.strip()]
    df = decompose.load_raw(args.raw)
    fit = predict.fit_from_raw(df)
    table = predict.predict_table(df, ns, n_boot=args.boot, seed=args.seed)

    with __import__("pandas").option_context("display.float_format", lambda v: f"{v:.1f}"):
        print(table.to_string(index=False))
    print(
        f"\nUSL fit: lambda={fit.lam:.1f}, alpha={fit.alpha:.3f} (contention), "
        f"beta={fit.beta:.4f} (coherency)"
    )
    print(f"Measured concurrency {fit.c_min:g}-{fit.c_max:g}; outside that is extrapolation.")
    if math.isfinite(fit.knee):
        print(f"Predicted throughput-optimal concurrency (knee): {fit.knee:.1f}")
    else:
        print("No interior knee — throughput saturates without going retrograde in this range.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="morpheus", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check-gpu", help="detect GPU/driver or halt").set_defaults(func=cmd_check_gpu)

    sw = sub.add_parser("sweep", help="run the concurrency sweep")
    sw.add_argument("--model", required=True)
    sw.add_argument("--concurrency", default="1,2,4,8,16,32,48")
    sw.add_argument("--prompt-len", type=int, default=512, dest="prompt_len")
    sw.add_argument("--output-len", type=int, default=256, dest="output_len")
    sw.add_argument("--n-requests", type=int, default=256, dest="n_requests")
    sw.add_argument("--dtype", default="auto")
    sw.add_argument("--max-model-len", type=int, default=None, dest="max_model_len")
    sw.add_argument("--gpu-mem-util", type=float, default=0.90, dest="gpu_mem_util")
    sw.add_argument("--seed", type=int, default=0)
    sw.add_argument("--port", type=int, default=8000)
    sw.add_argument("--raw", default="results/raw")
    sw.add_argument("--arrival-rate", default="", dest="arrival_rate",
                    help="comma-separated req/s (e.g. '0.5,1,2,4'): run an open-loop "
                         "Poisson sweep instead of the closed-loop concurrency sweep")
    sw.add_argument("--enforce-eager", action="store_true", dest="enforce_eager",
                    help="pass --enforce-eager to vLLM (skip CUDA-graph capture). "
                         "Needed on older GPUs like the T4 (compute cap 7.5) where the "
                         "graph/FlashAttention path is unstable over long runs.")
    sw.add_argument("--no-launch", action="store_true",
                    help="assume a server is already running (don't spawn vLLM)")
    sw.add_argument("--yes", "-y", action="store_true", help="skip the run-plan confirmation")
    sw.set_defaults(func=cmd_sweep)

    an = sub.add_parser("analyze", help="print derived sweep table")
    an.add_argument("--raw", default="results/raw")
    an.add_argument("--mode", choices=("closed", "open"), default="closed",
                    help="closed = concurrency sweep (raw_c*); open = Poisson rate sweep (raw_r*)")
    an.set_defaults(func=cmd_analyze)

    pl = sub.add_parser("plot", help="regenerate figures")
    pl.add_argument("--raw", default="results/raw")
    pl.add_argument("--figures", default="results/figures")
    pl.set_defaults(func=cmd_plot)

    pr = sub.add_parser(
        "predict",
        help="predict throughput at un-run concurrencies (USL fit + bootstrap band, "
             "tagged INTERPOLATED / EXTRAPOLATED)",
    )
    pr.add_argument("--raw", default="results/raw")
    pr.add_argument("--at", required=True,
                    help="comma-separated concurrencies to predict, e.g. '3,6,12,24'")
    pr.add_argument("--boot", type=int, default=1000,
                    help="bootstrap resamples for the confidence band (default 1000)")
    pr.add_argument("--seed", type=int, default=0)
    pr.set_defaults(func=cmd_predict)

    ct = sub.add_parser(
        "certify",
        help="verdict per series: TRUSTED / UNDERPOWERED / NONSTATIONARY "
             "(exit 0 only if all TRUSTED — usable as a CI gate)",
    )
    ct.add_argument("--raw", default="results/raw")
    ct.add_argument("--mode", choices=("closed", "open"), default="closed",
                    help="closed = concurrency sweep (raw_c*); open = Poisson rate sweep (raw_r*)")
    ct.add_argument("--file", default="",
                    help="certify another harness's export instead: JSON array or "
                         "single-column CSV of per-request latencies (ms), in time order")
    ct.add_argument("--rel-ci", type=float, default=0.05, dest="rel_ci",
                    help="target relative 95%% CI half-width on the mean (default 0.05)")
    ct.set_defaults(func=cmd_certify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
