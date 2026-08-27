#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="shu_p4_3seed_v1"
EXISTING_SCALE_LOG_DIR="experiments/logs/shu_amplitude_correction_v1/shu_mi_scale_075_125_p4_seed3407/vision/shu_mi"

run_case() {
  local seed="$1"
  local amplitude_scale="$2"
  local tag="$3"
  local run_name="shu_mi_p4_${tag}_seed${seed}"

  echo "[$(date -Is)] START dataset=SHU-MI P=4 tag=${tag} seed=${seed}"
  bash experiments/run_downstream.sh \
    --dataset SHU-MI \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${run_name}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/${run_name}" \
    --device cuda \
    --cuda 0 \
    --seed "${seed}" \
    --vision_fold_factor 4 \
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
  echo "[$(date -Is)] DONE dataset=SHU-MI P=4 tag=${tag} seed=${seed}"
}

best_val_pr() {
  local log_file="$1"
  sed -nE 's/.*Val Evaluation:.*pr_auc: ([0-9.]+).*/\1/p' "${log_file}" \
    | sort -nr \
    | head -n 1
}

# Complete the missing matched control first.
run_case 3407 false no_scale

no_scale_log="$(find "experiments/logs/${EXPERIMENT_NAME}/shu_mi_p4_no_scale_seed3407" -type f -name '*.log' -print -quit)"
scale_log="$(find "${EXISTING_SCALE_LOG_DIR}" -type f -name '*.log' -print -quit)"
no_scale_val="$(best_val_pr "${no_scale_log}")"
scale_val="$(best_val_pr "${scale_log}")"

if awk -v no_scale="${no_scale_val}" -v scale="${scale_val}" 'BEGIN { exit !(no_scale >= scale) }'; then
  selected_scale=false
  selected_tag=no_scale
else
  selected_scale=true
  selected_tag=scale_075_125
fi

echo "[$(date -Is)] SELECT P=4 no_scale_val_pr=${no_scale_val} scale_val_pr=${scale_val} selected=${selected_tag}"

for seed in 3408 3409; do
  run_case "${seed}" "${selected_scale}" "${selected_tag}"
done

echo "[$(date -Is)] ALL DONE"
