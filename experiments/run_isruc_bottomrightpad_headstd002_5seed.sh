#!/usr/bin/env bash
set -euo pipefail

# ISRUC five-seed confirmation with bottom/right-only padding and a smaller
# truncated-normal classifier-head initialization. All other hyperparameters
# come from the ISRUC entry in configs/downstream.py.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-isruc_p12_bottomrightpad_headstd002_5seed_v1}"
PYTHON_BIN="${PYTHON_BIN:-/home/netease/miniconda3/envs/eeg/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
SEEDS="${SEEDS:-42 43 44 45 46}"

for seed in ${SEEDS}; do
  echo "[$(date -Is)] START ISRUC bottom/right padding head-std=.002 seed=${seed}"
  bash experiments/run_downstream.sh \
    --dataset ISRUC \
    --python "${PYTHON_BIN}" \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/seed${seed}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/seed${seed}" \
    --device cuda --cuda "${CUDA_ID}" --seed "${seed}" \
    --vision_head_init_std 0.002
  echo "[$(date -Is)] DONE ISRUC bottom/right padding head-std=.002 seed=${seed}"
done
