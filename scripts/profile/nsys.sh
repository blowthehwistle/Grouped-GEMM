#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

nsys profile -t cuda,cublas,cusparse --gpu-metrics-device all --cudabacktrace=all --cuda-memory-usage true ./bin/tmain_4k "$1" "$2"
#nsys profile -t cuda,cublas,cusparse --gpu-metrics-device all --cudabacktrace=all --cuda-memory-usage true ./cutensor