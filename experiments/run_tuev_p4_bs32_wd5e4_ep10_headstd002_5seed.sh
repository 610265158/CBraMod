#!/usr/bin/env bash
set -euo pipefail

# TUEV five-seed batch-size-32, epoch-10 ablation.
# Keeps the previously-explored non-canonical recipe (wd=5e-4, ls=0, clip=1,
# head_init_std=0.002) frozen, and ONLY reverts batch_size 64->32 and
# epochs/early_stop 30->10 versus tuev_p4_bs64_lr1e3_wd5e4_ep30_ls0_clip1_headstd002_5seed_v1.
# Purpose: isolate whether bs/ep were the drivers of the high population std
# and lower kappa seen in the bs64/ep30 run.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-tuev_p4_bs32_lr1e3_wd5e4_ep10_ls0_clip1_headstd002_5seed_v1}"
PYTHON_BIN="${PYTHON_BIN:-/home/netease/miniconda3/envs/eeg/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
SEEDS="${SEEDS:-42 43 44 45 46}"

for seed in ${SEEDS}; do
  echo "[$(date -Is)] START TUEV P=4 bs=32 ep=10 wd=5e-4 ls=0 clip=1 head-std=.002 seed=${seed}"
  bash experiments/run_downstream.sh \
    --dataset TUEV \
    --python "${PYTHON_BIN}" \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/seed${seed}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/seed${seed}" \
    --device cuda --cuda "${CUDA_ID}" --seed "${seed}" \
    --backbone_name efficientnet_b0 \
    --vision_fold_factor 4 \
    --vision_head_init_std 0.002 \
    --use_pretrained_weights true \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --batch_size 32 \
    --num_workers 4 \
    --epochs 10 \
    --early_stop 10 \
    --min_lr 1e-6 \
    --warmup_epochs 0 \
    --warmup_start_factor 0.1 \
    --clip_value 1 \
    --ema_decay 0 \
    --optimizer AdamW \
    --label_smoothing 0 \
    --dropout 0.1 \
    --drop_path_rate 0 \
    --multi_lr false \
    --balanced_sampling false \
    --mirror_augmentation false \
    --time_roll_augmentation false \
    --amplitude_scale_augmentation false \
    --mixup_augmentation false \
    --amp true \
    --amp_dtype bfloat16 \
    --selection_metric kappa \
    --test_each_epoch false \
    --run_final_test true
  echo "[$(date -Is)] DONE TUEV P=4 bs=32 ep=10 wd=5e-4 ls=0 clip=1 head-std=.002 seed=${seed}"
done
