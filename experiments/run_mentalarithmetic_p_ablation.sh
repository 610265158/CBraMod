#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CUDA_ID="${CUDA_ID:-0}"

bash experiments/run_downstream.sh --config configs/ablation_p/MentalArithmetic_P1.yaml --device cuda --cuda "${CUDA_ID}"
bash experiments/run_downstream.sh --config configs/ablation_p/MentalArithmetic_P2.yaml --device cuda --cuda "${CUDA_ID}"
bash experiments/run_downstream.sh --config configs/ablation_p/MentalArithmetic_P4.yaml --device cuda --cuda "${CUDA_ID}"
bash experiments/run_downstream.sh --config configs/ablation_p/MentalArithmetic_P8.yaml --device cuda --cuda "${CUDA_ID}"
