#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CUDA_ID="${CUDA_ID:-0}"

bash experiments/run_downstream.sh --config configs/backbones/convnext_tiny_dinov3/TUEV_random_init.yaml --device cuda --cuda "${CUDA_ID}"
bash experiments/run_downstream.sh --config configs/backbones/convnext_tiny_dinov3/PhysioNet-MI_random_init.yaml --device cuda --cuda "${CUDA_ID}"
bash experiments/run_downstream.sh --config configs/backbones/vit_small_dinov3/TUEV_random_init.yaml --device cuda --cuda "${CUDA_ID}"
bash experiments/run_downstream.sh --config configs/backbones/vit_small_dinov3/PhysioNet-MI_random_init.yaml --device cuda --cuda "${CUDA_ID}"
