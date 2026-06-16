"""Command-line entry point: ``decodebound <subcommand>``.

Subcommands:
  check-gpu   detect the GPU/driver, or print the halt message and exit non-zero
  sweep       launch vLLM, run the concurrency sweep, write results/raw/ + run_meta
  analyze     load raw data, print the derived sweep table + the knee
  plot        regenerate the three figures from raw data
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
            "No CUDA GPU detected. DecodeBound will not fall back to CPU.\n"
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
    )
    wl = workload.Workload(
        name=f"{args.prompt_len}in-{args.output_len}out",
        prompt_len=args.prompt_len,
        output_len=args.output_len,
        n_requests=args.n_requests,
        seed=args.seed,
    )
    concurrencies = _parse_concurrency(args.concurrency)

    n_points = len(concurrencies)
    total_requests = n_points * wl.n_requests
    print(f"Planned: {n_points} concurrency points x {wl.n_requests} requests "
          f"= {total_requests} requests against {args.model}.")
    if not args.yes:
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return 1

    meta = sweep.run_sweep(
        cfg, wl, concurrencies, out_dir=args.raw, launch_server=not args.no_launch
    )
    print(f"Wrote {meta}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from analysis import decompose

    df = decompose.load_raw(args.raw)
    agg = decompose.aggregate_sweep(df)
    knee = decompose.find_knee(agg)
    cols = ["concurrency", "n_requests", "throughput_tok_s",
            "itl_p50", "itl_p99", "ttft_p50", "ttft_p99", "itl_lag1_acf"]
    with __import__("pandas").option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(agg[cols].to_string(index=False))
    print(f"\nKnee (honest operating point): concurrency = {knee}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from analysis import plots

    paths = plots.generate_all(raw_dir=args.raw, fig_dir=args.figures)
    for p in paths:
        print(f"Wrote {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="decodebound", description=__doc__)
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
    sw.add_argument("--no-launch", action="store_true",
                    help="assume a server is already running (don't spawn vLLM)")
    sw.add_argument("--yes", "-y", action="store_true", help="skip the run-plan confirmation")
    sw.set_defaults(func=cmd_sweep)

    an = sub.add_parser("analyze", help="print derived sweep table")
    an.add_argument("--raw", default="results/raw")
    an.set_defaults(func=cmd_analyze)

    pl = sub.add_parser("plot", help="regenerate figures")
    pl.add_argument("--raw", default="results/raw")
    pl.add_argument("--figures", default="results/figures")
    pl.set_defaults(func=cmd_plot)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
