#!/usr/bin/env bash
set -euo pipefail

# Final BCIC2020-3 five-seed run. Hyperparameters and folding geometry come
# from FINALIZED_FIVE_SEED_RECIPES in configs/downstream.py.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-bcic2020_3_b0_p1_bottomrightpad_5seed_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_ID="${CUDA_ID:-0}"
SEEDS="${SEEDS:-42 43 44 45 46}"

for seed in ${SEEDS}; do
  echo "[$(date -Is)] START BCIC2020-3 bottom/right padding seed=${seed}"
  bash experiments/run_downstream.sh \
    --dataset BCIC2020-3 \
    --python "${PYTHON_BIN}" \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/seed${seed}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/seed${seed}" \
    --device cuda --cuda "${CUDA_ID}" --seed "${seed}"
  echo "[$(date -Is)] DONE BCIC2020-3 bottom/right padding seed=${seed}"
done
