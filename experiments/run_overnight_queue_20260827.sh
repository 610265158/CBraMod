#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CUDA_ID="${CUDA_ID:-0}"

CUDA_ID="${CUDA_ID}" bash experiments/run_shu_b0_b5_3seed.sh
CUDA_ID="${CUDA_ID}" bash experiments/run_fixed_recipe_reruns.sh \
  BCIC2020-3 Mumtaz2016 MentalArithmetic

echo "[$(date -Is)] FULL OVERNIGHT QUEUE DONE"
