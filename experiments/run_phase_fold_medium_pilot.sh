#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="phase_fold_medium_pilot_v1"
SEED=3407

run_case() {
  local dataset="$1"
  local fold_factor="$2"
  local epochs="$3"
  local lr="$4"
  local weight_decay="$5"
  local metric="$6"
  local amplitude_scale="$7"
  local run_name="${dataset,,}_p${fold_factor}_seed${SEED}"
  run_name="${run_name//-/_}"

  echo "[$(date -Is)] START dataset=${dataset} P=${fold_factor} seed=${SEED} epochs=${epochs} lr=${lr} wd=${weight_decay} amplitude_scale=${amplitude_scale}"
  bash experiments/run_downstream.sh \
    --dataset "${dataset}" \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${run_name}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/${run_name}" \
    --device cuda \
    --cuda 0 \
    --seed "${SEED}" \
    --vision_fold_factor "${fold_factor}" \
    --backbone_name efficientnet_b0 \
    --use_pretrained_weights true \
    --epochs "${epochs}" \
    --batch_size 32 \
    --num_workers 4 \
    --lr "${lr}" \
    --weight_decay "${weight_decay}" \
    --multi_lr false \
    --dropout 0.1 \
    --early_stop "${epochs}" \
    --amplitude_scale_augmentation "${amplitude_scale}" \
    --amplitude_scale_prob 0.5 \
    --amplitude_scale_min 0.25 \
    --amplitude_scale_max 1.25 \
    --amp true \
    --amp_dtype float16 \
    --test_each_epoch false \
    --run_final_test true \
    --selection_metric "${metric}"
  echo "[$(date -Is)] DONE dataset=${dataset} P=${fold_factor} seed=${SEED}"
}

# FACED: P=1 retains 32 rows; P=2 exposes two vertical backbone cells.
run_case FACED 1 50 1e-3 5e-3 kappa false
run_case FACED 2 50 1e-3 5e-3 kappa false

# PhysioNet-MI: the unfolded 64-channel geometry is already CNN-friendly.
run_case PhysioNet-MI 1 30 2e-3 5e-3 kappa false
run_case PhysioNet-MI 2 30 2e-3 5e-3 kappa false

# SHU-MI keeps its documented clip-512/no-divisor loader and amplitude scaling.
run_case SHU-MI 2 20 1e-3 5e-3 pr_auc true
run_case SHU-MI 4 20 1e-3 5e-3 pr_auc true

echo "[$(date -Is)] ALL DONE"
