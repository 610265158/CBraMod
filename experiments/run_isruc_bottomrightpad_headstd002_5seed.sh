#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CONFIG="configs/backbones/efficientnet_b0/ISRUC.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_ID="${CUDA_ID:-0}"

exec bash experiments/run_downstream.sh --config "${CONFIG}" --python "${PYTHON_BIN}" \
  --device cuda --cuda "${CUDA_ID}"
