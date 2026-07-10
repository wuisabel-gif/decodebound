#!/usr/bin/env bash
# Regenerate every committed result and figure from scratch.
#
# Usage:
#   ./reproduce.sh --model meta-llama/Llama-3.1-8B-Instruct [--concurrency 1,2,4,8,16,32,48]
#
# Halts immediately if no CUDA GPU is present — DecodeBound does not measure CPU.
set -euo pipefail

MODEL=""
CONCURRENCY="1,2,4,8,16,32,48"
BACKEND="vllm"
SWEEP="concurrency"
RAW="results/raw"
FIG="results/figures"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --backend)     BACKEND="$2"; shift 2 ;;
    --sweep)       SWEEP="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "error: --model is required (e.g. --model meta-llama/Llama-3.1-8B-Instruct)" >&2
  exit 2
fi

echo "==> Checking for a CUDA GPU"
morpheus check-gpu   # exits non-zero (and set -e halts) if no GPU

echo "==> Sweep ($SWEEP) on $MODEL via $BACKEND: $CONCURRENCY"
morpheus sweep --model "$MODEL" --concurrency "$CONCURRENCY" --raw "$RAW" --yes

echo "==> Regenerating figures"
morpheus plot --raw "$RAW" --figures "$FIG"

echo "==> Derived sweep table"
morpheus analyze --raw "$RAW"

echo "Done. Raw data in $RAW (with run_meta.json); figures in $FIG."
