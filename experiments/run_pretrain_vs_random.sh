#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CUDA_ID="${CUDA_ID:-0}"

echo "[pretrain_vs_random] TUEV random-init (5 seeds) start: $(date)"
bash experiments/run_downstream.sh --config configs/backbones/efficientnet_b0/TUEV_random_init.yaml \
  --device cuda --cuda "${CUDA_ID}"

echo "[pretrain_vs_random] PhysioNet-MI random-init (5 seeds) start: $(date)"
bash experiments/run_downstream.sh --config configs/backbones/efficientnet_b0/PhysioNet-MI_random_init.yaml \
  --device cuda --cuda "${CUDA_ID}"

echo "[pretrain_vs_random] ALL DONE: $(date)"
