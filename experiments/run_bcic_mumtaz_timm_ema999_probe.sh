#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="bcic_mumtaz_timm_ema999_bf16_seed3407_v1"
CUDA_ID="${CUDA_ID:-0}"

run_case() {
  local dataset="$1" lr="$2" weight_decay="$3" metric="$4"
  local run_name="${dataset,,}_p2_seed3407"
  run_name="${run_name//-/_}"

  echo "[$(date -Is)] START dataset=${dataset} P=2 timm_EMA=0.999"
  bash experiments/run_downstream.sh \
    --dataset "${dataset}" \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${run_name}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/${run_name}" \
    --device cuda --cuda "${CUDA_ID}" --seed 3407 \
    --backbone_name efficientnet_b0 --vision_fold_factor 2 \
    --use_pretrained_weights true \
    --epochs 30 --batch_size 32 --num_workers 4 \
    --lr "${lr}" --weight_decay "${weight_decay}" \
    --optimizer AdamW --label_smoothing 0.1 --dropout 0.1 --multi_lr false \
    --warmup_epochs 3 --warmup_start_factor 0.1 \
    --clip_value 1 --amp true --amp_dtype bfloat16 --ema_decay 0.999 \
    --mirror_augmentation false --time_roll_augmentation false \
    --amplitude_scale_augmentation false \
    --early_stop 30 --test_each_epoch false --run_final_test true \
    --selection_metric "${metric}"
  echo "[$(date -Is)] DONE dataset=${dataset} P=2 timm_EMA=0.999"
}

run_case BCIC2020-3 1e-3 5e-3 kappa
run_case Mumtaz2016 5e-4 5e-2 pr_auc

echo "[$(date -Is)] TIMM EMA PROBE DONE"
