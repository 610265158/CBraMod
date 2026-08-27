#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
EXPERIMENT_NAME="physio_p2_lr1e3_3seed_v1"

for seed in 3407 3408 3409; do
  echo "[$(date -Is)] START PhysioNet-MI P=2 lr=1e-3 seed=${seed}"
  bash experiments/run_downstream.sh \
    --dataset PhysioNet-MI \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/seed${seed}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/seed${seed}" \
    --device cuda \
    --cuda 0 \
    --seed "${seed}" \
    --vision_fold_factor 2 \
    --backbone_name efficientnet_b0 \
    --use_pretrained_weights true \
    --epochs 30 \
    --batch_size 32 \
    --num_workers 4 \
    --lr 1e-3 \
    --weight_decay 5e-3 \
    --clip_value -1 \
    --multi_lr false \
    --dropout 0.1 \
    --early_stop 30 \
    --mirror_augmentation false \
    --mirror_prob 0.5 \
    --time_roll_augmentation false \
    --amplitude_scale_augmentation false \
    --amp true \
    --amp_dtype float16 \
    --test_each_epoch false \
    --run_final_test true \
    --selection_metric kappa
  echo "[$(date -Is)] DONE PhysioNet-MI P=2 lr=1e-3 seed=${seed}"
done

echo "[$(date -Is)] ALL DONE"
