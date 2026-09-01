#!/usr/bin/env bash
set -euo pipefail

# PhysioNet-MI reporting run, seeds 42--46.
# Use one LR for the complete network: multi_lr=false means
# backbone_lr_scale is intentionally inactive and all parameters use 2e-3.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-physionet_mi_p1_lr2e3_singlelr_wd5e3_warm3_ema995_5seed_v1}"
PYTHON_BIN="${PYTHON_BIN:-/home/netease/miniconda3/envs/eeg/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
SEEDS="${SEEDS:-42 43 44 45 46}"

for seed in ${SEEDS}; do
  echo "[$(date -Is)] START PhysioNet-MI single-LR seed=${seed}"
  bash experiments/run_downstream.sh \
    --dataset PhysioNet-MI \
    --python "${PYTHON_BIN}" \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/seed${seed}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/seed${seed}" \
    --device cuda --cuda "${CUDA_ID}" --seed "${seed}" \
    --backbone_name efficientnet_b0 \
    --vision_fold_factor 1 \
    --use_pretrained_weights true \
    --lr 2e-3 \
    --backbone_lr_scale 0.1 \
    --multi_lr false \
    --weight_decay 5e-3 \
    --optimizer AdamW \
    --min_lr 1e-6 \
    --batch_size 32 \
    --num_workers 4 \
    --epochs 30 \
    --warmup_epochs 3 \
    --warmup_start_factor 0.1 \
    --ema_decay 0.995 \
    --clip_value -1 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --drop_path_rate 0 \
    --balanced_sampling false \
    --mirror_augmentation false \
    --time_roll_augmentation false \
    --amplitude_scale_augmentation false \
    --mixup_augmentation false \
    --amp true \
    --amp_dtype bfloat16 \
    --early_stop 10 \
    --selection_metric kappa \
    --test_each_epoch false \
    --run_final_test true
  echo "[$(date -Is)] DONE PhysioNet-MI single-LR seed=${seed}"
done
