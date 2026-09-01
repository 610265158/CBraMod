#!/usr/bin/env bash
set -euo pipefail

# Final MentalArithmetic five-seed run. Hyperparameters, preprocessing, head
# initialization, and folding geometry come from FINALIZED_FIVE_SEED_RECIPES.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-mentalarithmetic_p4_headstd002_ema995_5seed_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_ID="${CUDA_ID:-0}"
SEEDS="${SEEDS:-42 43 44 45 46}"

for seed in ${SEEDS}; do
  echo "[$(date -Is)] START MentalArithmetic head-std=.002 EMA=.995 seed=${seed}"
  bash experiments/run_downstream.sh \
    --dataset MentalArithmetic \
    --python "${PYTHON_BIN}" \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/seed${seed}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/seed${seed}" \
    --device cuda --cuda "${CUDA_ID}" --seed "${seed}"
  echo "[$(date -Is)] DONE MentalArithmetic head-std=.002 EMA=.995 seed=${seed}"
done
