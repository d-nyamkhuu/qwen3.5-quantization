#!/usr/bin/env bash
# Load-test the server on localhost:8000 at each concurrency while sampling GPU memory and power.
#   bash bench/sweep.sh <out_dir> [concurrency ...]      default concurrencies: 1 8 32 128 256
# Run it from the repository root. Extra `vllm bench serve` flags go in BENCH_ARGS (e.g. --save-detailed).
set -euo pipefail
OUT=$1
shift
CONCURRENCY=${*:-1 8 32 128 256}
IMAGE=vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e

mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits \
  -lms 500 > "$OUT/gpu_samples.csv" &
SAMPLER=$!
trap 'kill $SAMPLER' EXIT

for c in $CONCURRENCY; do
  prompts=$(( c * 8 < 32 ? 32 : c * 8 > 1024 ? 1024 : c * 8 ))  # 8 per slot, between 32 and 1,024
  echo "$(date +%s) start c=$c" >> "$OUT/phases.txt"
  docker run --rm --network host -v "$PWD":/work -w /work --entrypoint vllm "$IMAGE" bench serve \
    --backend openai --base-url http://localhost:8000 --endpoint /v1/completions --model m \
    --tokenizer data/models/Qwen3.5-9B --dataset-name random --random-input-len 1024 --random-output-len 512 \
    --ignore-eos --num-prompts "$prompts" --max-concurrency "$c" --num-warmups $(( c < 8 ? 8 : c )) --seed 42 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "$OUT" --result-filename "c$c.json" ${BENCH_ARGS:-}
  echo "$(date +%s) end c=$c" >> "$OUT/phases.txt"
done
