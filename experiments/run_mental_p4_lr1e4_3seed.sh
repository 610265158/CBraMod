#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
EXPERIMENT_NAME="mental_p4_lr1e4_3seed_v1"

for seed in 3407 3408 3409; do
  echo "[$(date -Is)] START MentalArithmetic P=4 lr=1e-4 roc_auc seed=${seed}"
  bash experiments/run_downstream.sh \
    --dataset MentalArithmetic \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/seed${seed}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/seed${seed}" \
    --device cuda \
    --cuda 0 \
    --seed "${seed}" \
    --vision_fold_factor 4 \
    --backbone_name efficientnet_b0 \
    --use_pretrained_weights true \
    --epochs 10 \
    --batch_size 32 \
    --num_workers 4 \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --clip_value -1 \
    --multi_lr false \
    --dropout 0.1 \
    --early_stop 10 \
    --mirror_augmentation false \
    --time_roll_augmentation false \
    --amplitude_scale_augmentation false \
    --amp true \
    --amp_dtype float16 \
    --test_each_epoch false \
    --run_final_test true \
    --selection_metric roc_auc
  echo "[$(date -Is)] DONE MentalArithmetic P=4 lr=1e-4 roc_auc seed=${seed}"
done

echo "[$(date -Is)] ALL DONE"
