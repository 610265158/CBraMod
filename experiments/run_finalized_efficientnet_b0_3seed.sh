#!/usr/bin/env bash
set -euo pipefail

# Reproduce the finalized EfficientNet-B0 recipes in PHASE_FOLD_RESULTS.md.
# With no positional arguments all 11 datasets are run. To run a subset:
#   bash experiments/run_finalized_efficientnet_b0_3seed.sh TUEV ISRUC

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="finalized_efficientnet_b0_3seed_v1"
CUDA_ID="${CUDA_ID:-0}"
REQUESTED=("$@")

is_requested() {
  local dataset="$1"
  if [[ ${#REQUESTED[@]} -eq 0 ]]; then
    return 0
  fi
  local requested
  for requested in "${REQUESTED[@]}"; do
    if [[ "${requested}" == "${dataset}" ]]; then
      return 0
    fi
  done
  return 1
}

run_recipe() {
  local dataset="$1"
  local fold_factor="$2"
  local epochs="$3"
  local batch_size="$4"
  local lr="$5"
  local weight_decay="$6"
  local metric="$7"
  local clip_value="$8"
  local mirror="$9"
  local shu_scale="${10}"
  local warmup_epochs=0
  local ema_decay=0
  local binary_pos_weight=1
  local time_roll=false
  local time_roll_prob=1
  local time_roll_max_fraction=0.5
  local amp_dtype=float16
  local early_stop="${epochs}"

  if [[ "${dataset}" == "ISRUC" ]]; then
    warmup_epochs=3
    ema_decay=0.995
    time_roll=true
    time_roll_prob=0.5
    time_roll_max_fraction=0.25
  elif [[ "${dataset}" == "PhysioNet-MI" ]]; then
    warmup_epochs=3
    ema_decay=0.995
    early_stop=10
  elif [[ "${dataset}" == "MentalArithmetic" ]]; then
    warmup_epochs=3
    ema_decay=0.99
    binary_pos_weight=3
    time_roll=true
    time_roll_prob=0.5
    time_roll_max_fraction=0.25
    amp_dtype=bfloat16
  fi

  if ! is_requested "${dataset}"; then
    return
  fi

  local seed run_name
  for seed in 3407 3408 3409; do
    run_name="${dataset,,}_p${fold_factor}_seed${seed}"
    run_name="${run_name//-/_}"
    echo "[$(date -Is)] START dataset=${dataset} P=${fold_factor} seed=${seed}"
    bash experiments/run_downstream.sh \
      --dataset "${dataset}" \
      --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${run_name}" \
      --log_root "experiments/logs/${EXPERIMENT_NAME}/${run_name}" \
      --device cuda \
      --cuda "${CUDA_ID}" \
      --seed "${seed}" \
      --backbone_name efficientnet_b0 \
      --vision_fold_factor "${fold_factor}" \
      --use_pretrained_weights true \
      --epochs "${epochs}" \
      --batch_size "${batch_size}" \
      --num_workers 4 \
      --lr "${lr}" \
      --backbone_lr_scale 0.1 \
      --weight_decay "${weight_decay}" \
      --clip_value "${clip_value}" \
      --warmup_epochs "${warmup_epochs}" \
      --warmup_start_factor 0.1 \
      --ema_decay "${ema_decay}" \
      --optimizer AdamW \
      --label_smoothing 0.1 \
      --binary_pos_weight "${binary_pos_weight}" \
      --dropout 0.1 \
      --drop_path_rate 0 \
      --multi_lr false \
      --early_stop "${early_stop}" \
      --balanced_sampling false \
      --mirror_augmentation "${mirror}" \
      --mirror_prob 0.5 \
      --time_roll_augmentation "${time_roll}" \
      --time_roll_prob "${time_roll_prob}" \
      --time_roll_max_fraction "${time_roll_max_fraction}" \
      --amplitude_scale_augmentation false \
      --shu_scale "${shu_scale}" \
      --amp true \
      --amp_dtype "${amp_dtype}" \
      --test_each_epoch false \
      --run_final_test true \
      --selection_metric "${metric}"
    echo "[$(date -Is)] DONE dataset=${dataset} P=${fold_factor} seed=${seed}"
  done
}

# dataset P epochs batch lr wd metric clip mirror shu_scale
run_recipe CHB-MIT          2 10 32 1e-3 5e-3 pr_auc -1 false 64
run_recipe TUAB             2  5 32 1e-3 5e-4 pr_auc  1 false 64
run_recipe TUEV             4 10 32 1e-3 5e-3 kappa  -1 false 64
run_recipe ISRUC           12 15 16 1e-3 5e-3 kappa  -1 true  64
run_recipe FACED            2 50 32 1e-3 5e-3 kappa  -1 false 64
run_recipe SEED-V           8 50 32 5e-4 5e-3 kappa  -1 false 64
run_recipe PhysioNet-MI     1 30 32 2e-3 5e-3 kappa  -1 false 64
run_recipe SHU-MI           4 20 32 1e-3 5e-3 pr_auc -1 false 64
run_recipe BCIC2020-3       1 30 32 1e-3 5e-3 kappa  -1 false 64
run_recipe Mumtaz2016       2 30 32 5e-4 5e-2 pr_auc -1 false 64
run_recipe MentalArithmetic 2 30 64 1e-3 5e-4 pr_auc -1 true 64

echo "[$(date -Is)] REQUESTED RECIPES DONE"
