#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

for seed in 3407 3409; do
  echo "[$(date -Is)] START MentalArithmetic P=4 BF16 lr=1e-3 seed=${seed}"
  bash experiments/run_downstream.sh \
    --dataset MentalArithmetic \
    --model_root "experiments/checkpoints/mental_p4_bf16_3seed_v1/seed${seed}" \
    --log_root "experiments/logs/mental_p4_bf16_3seed_v1/seed${seed}" \
    --device cuda \
    --cuda 0 \
    --seed "${seed}" \
    --vision_fold_factor 4 \
    --backbone_name efficientnet_b0 \
    --use_pretrained_weights true \
    --epochs 10 \
    --batch_size 32 \
    --num_workers 4 \
    --lr 1e-3 \
    --weight_decay 1e-2 \
    --clip_value -1 \
    --multi_lr false \
    --dropout 0.1 \
    --early_stop 10 \
    --mirror_augmentation false \
    --time_roll_augmentation false \
    --amplitude_scale_augmentation false \
    --amp true \
    --amp_dtype bfloat16 \
    --test_each_epoch false \
    --run_final_test true \
    --selection_metric pr_auc
  echo "[$(date -Is)] DONE MentalArithmetic P=4 BF16 lr=1e-3 seed=${seed}"
done

echo "[$(date -Is)] ALL DONE (seed 3408 already completed in mental_p4_bf16_probe_v1)"
