#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="shu_amplitude_correction_v1"
SEED=3407

run_case() {
  local fold_factor="$1"
  local amplitude_scale="$2"
  local tag="$3"
  local run_name="shu_mi_${tag}_p${fold_factor}_seed${SEED}"

  echo "[$(date -Is)] START dataset=SHU-MI tag=${tag} P=${fold_factor} seed=${SEED}"
  bash experiments/run_downstream.sh \
    --dataset SHU-MI \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${run_name}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/${run_name}" \
    --device cuda \
    --cuda 0 \
    --seed "${SEED}" \
    --vision_fold_factor "${fold_factor}" \
    --backbone_name efficientnet_b0 \
    --use_pretrained_weights true \
    --epochs 20 \
    --batch_size 32 \
    --num_workers 4 \
    --lr 1e-3 \
    --weight_decay 5e-3 \
    --multi_lr false \
    --dropout 0.1 \
    --early_stop 20 \
    --amplitude_scale_augmentation "${amplitude_scale}" \
    --amplitude_scale_prob 0.5 \
    --amplitude_scale_min 0.75 \
    --amplitude_scale_max 1.25 \
    --amp true \
    --amp_dtype float16 \
    --test_each_epoch false \
    --run_final_test true \
    --selection_metric pr_auc
  echo "[$(date -Is)] DONE dataset=SHU-MI tag=${tag} P=${fold_factor} seed=${SEED}"
}

# Matched no-augmentation control and corrected +/-25% runs.
run_case 2 false no_scale
run_case 2 true scale_075_125
run_case 4 true scale_075_125

echo "[$(date -Is)] ALL DONE"
