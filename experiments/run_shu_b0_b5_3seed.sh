#!/usr/bin/env bash
set -euo pipefail

# Controlled SHU-MI backbone comparison. Both backbones use the same normal
# BatchNorm behavior, phase fold, augmentations, optimizer, and reporting rule.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CUDA_ID="${CUDA_ID:-0}"
EXPERIMENT_NAME="shu_b0_b5_3seed_v1"

run_backbone() {
  local backbone="$1"
  local short_name="$2"
  local seed run_name

  for seed in 3407 3408 3409; do
    run_name="${short_name}_seed${seed}"
    echo "[$(date -Is)] START SHU-MI backbone=${backbone} seed=${seed}"
    bash experiments/run_downstream.sh \
      --dataset SHU-MI \
      --cuda "${CUDA_ID}" \
      --device cuda \
      --seed "${seed}" \
      --backbone_name "${backbone}" \
      --vision_fold_factor 2 \
      --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${run_name}" \
      --log_root "experiments/logs/${EXPERIMENT_NAME}/${run_name}" \
      --epochs 20 \
      --batch_size 32 \
      --num_workers 4 \
      --lr 1e-3 \
      --backbone_lr_scale 1 \
      --weight_decay 5e-4 \
      --min_lr 1e-6 \
      --warmup_epochs 3 \
      --warmup_start_factor .1 \
      --clip_value -1 \
      --ema_decay .995 \
      --optimizer AdamW \
      --label_smoothing .1 \
      --binary_pos_weight 1 \
      --dropout .1 \
      --drop_path_rate 0 \
      --early_stop 20 \
      --frozen false \
      --multi_lr false \
      --use_pretrained_weights true \
      --balanced_sampling false \
      --mirror_augmentation false \
      --time_roll_augmentation true \
      --time_roll_prob .5 \
      --time_roll_max_fraction .25 \
      --amplitude_scale_augmentation false \
      --amp true \
      --amp_dtype float16 \
      --shu_clip_limit 512 \
      --shu_scale 64 \
      --test_each_epoch false \
      --run_final_test true \
      --selection_metric pr_auc
    echo "[$(date -Is)] DONE SHU-MI backbone=${backbone} seed=${seed}"
  done
}

run_backbone efficientnet_b0 b0
run_backbone efficientnet_b5 b5

echo "[$(date -Is)] SHU-MI B0/B5 THREE-SEED COMPARISON DONE"
