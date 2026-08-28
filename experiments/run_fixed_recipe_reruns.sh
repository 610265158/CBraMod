#!/usr/bin/env bash
set -euo pipefail

# Clean 3-seed reproduction for recipes marked in RERUN_QUEUE.md.  Dataset
# hyperparameters come from configs/downstream.py; this script only isolates
# output paths and fixes the reporting seeds.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CUDA_ID="${CUDA_ID:-0}"
EXPERIMENT_NAME="fixed_recipe_rerun_v2"
ALL_DATASETS=(BCIC2020-3 Mumtaz2016 MentalArithmetic)
REQUESTED=("$@")

if [[ ${#REQUESTED[@]} -eq 0 ]]; then
  REQUESTED=("${ALL_DATASETS[@]}")
fi

is_marked_dataset() {
  local candidate="$1"
  local dataset
  for dataset in "${ALL_DATASETS[@]}"; do
    if [[ "${candidate}" == "${dataset}" ]]; then
      return 0
    fi
  done
  return 1
}

for dataset in "${REQUESTED[@]}"; do
  if ! is_marked_dataset "${dataset}"; then
    echo "Unknown rerun dataset: ${dataset}" >&2
    echo "Choose from: ${ALL_DATASETS[*]}" >&2
    exit 2
  fi

  for seed in 3407 3408 3409; do
    safe_dataset="${dataset,,}"
    safe_dataset="${safe_dataset//-/_}"
    run_name="${safe_dataset}_seed${seed}"
    echo "[$(date -Is)] START dataset=${dataset} seed=${seed} recipe=fixed_recipe_v2"
    bash experiments/run_downstream.sh \
      --dataset "${dataset}" \
      --cuda "${CUDA_ID}" \
      --device cuda \
      --seed "${seed}" \
      --backbone_name efficientnet_b0 \
      --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${run_name}" \
      --log_root "experiments/logs/${EXPERIMENT_NAME}/${run_name}"
    echo "[$(date -Is)] DONE dataset=${dataset} seed=${seed} recipe=fixed_recipe_v2"
  done
done

echo "[$(date -Is)] FIXED-RECIPE RERUNS DONE"
