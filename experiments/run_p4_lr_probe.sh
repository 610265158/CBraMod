#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
EXPERIMENT_NAME="p4_lr_probe_v1"

run_case() {
  local dataset="$1"
  local epochs="$2"
  local wd="$3"
  local lr="$4"
  local tag="$5"
  local name="${dataset,,}_p4_lr${tag}_seed3407"
  name="${name//-/_}"
  echo "[$(date -Is)] START dataset=${dataset} P=4 lr=${lr} seed=3407"
  bash experiments/run_downstream.sh \
    --dataset "${dataset}" \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${name}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/${name}" \
    --device cuda \
    --cuda 0 \
    --seed 3407 \
    --vision_fold_factor 4 \
    --backbone_name efficientnet_b0 \
    --use_pretrained_weights true \
    --epochs "${epochs}" \
    --batch_size 32 \
    --num_workers 4 \
    --lr "${lr}" \
    --weight_decay "${wd}" \
    --clip_value -1 \
    --multi_lr false \
    --dropout 0.1 \
    --early_stop "${epochs}" \
    --mirror_augmentation false \
    --time_roll_augmentation false \
    --amplitude_scale_augmentation false \
    --amp true \
    --amp_dtype float16 \
    --test_each_epoch false \
    --run_final_test true \
    --selection_metric pr_auc
  echo "[$(date -Is)] DONE dataset=${dataset} P=4 lr=${lr} seed=3407"
}

for lr in 1e-4 5e-4 1e-3; do
  run_case Mumtaz2016 30 5e-2 "${lr}" "${lr}"
  run_case MentalArithmetic 10 1e-2 "${lr}" "${lr}"
done

echo "[$(date -Is)] ALL DONE"
