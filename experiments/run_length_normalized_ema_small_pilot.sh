#!/usr/bin/env bash
set -euo pipefail

# Seed-3407 pilot for the length-normalized folding rule:
#   T < 1000 -> P=2, T=1000 -> P=2, T=2000 -> P=4, T=6000 -> P=12.
# This first stage uses three small datasets, all with P=2.  Other optimization
# settings remain at their finalized EfficientNet-B0 values so that the pilot
# isolates the new stability recipe as closely as possible.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="length_norm_p_ema999_bf16_small_seed3407_v1"
CUDA_ID="${CUDA_ID:-0}"
SEED=3407

run_case() {
  local dataset="$1"
  local epochs="$2"
  local lr="$3"
  local weight_decay="$4"
  local warmup_epochs="$5"
  local metric="$6"
  local run_name="${dataset,,}_p2_seed${SEED}"
  run_name="${run_name//-/_}"

  echo "[$(date -Is)] START dataset=${dataset} P=2 seed=${SEED} warmup=${warmup_epochs} bf16 EMA=0.999"
  bash experiments/run_downstream.sh \
    --dataset "${dataset}" \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${run_name}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/${run_name}" \
    --device cuda --cuda "${CUDA_ID}" --seed "${SEED}" \
    --backbone_name efficientnet_b0 --vision_fold_factor 2 \
    --use_pretrained_weights true \
    --epochs "${epochs}" --batch_size 32 --num_workers 4 \
    --lr "${lr}" --weight_decay "${weight_decay}" \
    --optimizer AdamW --label_smoothing 0.1 --dropout 0.1 \
    --warmup_epochs "${warmup_epochs}" --warmup_start_factor 0.1 \
    --clip_value 1 --amp true --amp_dtype bfloat16 --ema_decay 0.999 \
    --multi_lr false --time_roll_augmentation false \
    --amplitude_scale_augmentation false --mirror_augmentation false \
    --early_stop "${epochs}" --test_each_epoch false --run_final_test true \
    --selection_metric "${metric}"
  echo "[$(date -Is)] DONE dataset=${dataset} P=2 seed=${SEED}"
}

# dataset epochs lr wd warmup selection_metric
run_case BCIC2020-3       30 1e-3 5e-3 3 kappa
run_case Mumtaz2016       30 5e-4 5e-2 3 pr_auc
run_case MentalArithmetic 10 5e-4 1e-2 1 pr_auc

echo "[$(date -Is)] SMALL PILOT DONE"
