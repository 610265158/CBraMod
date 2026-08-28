#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="mental_p2_timm_ema99_bf16_seed3407_v1"

bash experiments/run_downstream.sh \
  --dataset MentalArithmetic \
  --model_root "experiments/checkpoints/${EXPERIMENT_NAME}" \
  --log_root "experiments/logs/${EXPERIMENT_NAME}" \
  --device cuda --cuda "${CUDA_ID:-0}" --seed 3407 \
  --backbone_name efficientnet_b0 --vision_fold_factor 2 \
  --use_pretrained_weights true \
  --epochs 10 --batch_size 32 --num_workers 4 \
  --lr 5e-4 --weight_decay 1e-2 --optimizer AdamW \
  --label_smoothing 0.1 --dropout 0.1 --multi_lr false \
  --warmup_epochs 1 --warmup_start_factor 0.1 \
  --clip_value 1 --amp true --amp_dtype bfloat16 --ema_decay 0.99 \
  --mirror_augmentation false --time_roll_augmentation false \
  --amplitude_scale_augmentation false \
  --early_stop 10 --test_each_epoch false --run_final_test true \
  --selection_metric pr_auc
