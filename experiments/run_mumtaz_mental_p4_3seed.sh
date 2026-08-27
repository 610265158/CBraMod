#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
EXPERIMENT_NAME="p4_mumtaz_mental_3seed_v1"

run_case() {
  local dataset="$1"
  local fold="$2"
  local epochs="$3"
  local lr="$4"
  local wd="$5"
  local seed="$6"
  local name="${dataset,,}_p${fold}_seed${seed}"
  name="${name//-/_}"
  echo "[$(date -Is)] START dataset=${dataset} P=${fold} seed=${seed}"
  bash experiments/run_downstream.sh \
    --dataset "${dataset}" \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${name}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/${name}" \
    --device cuda \
    --cuda 0 \
    --seed "${seed}" \
    --vision_fold_factor "${fold}" \
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
  echo "[$(date -Is)] DONE dataset=${dataset} P=${fold} seed=${seed}"
}

for seed in 3407 3408 3409; do
  run_case Mumtaz2016 4 30 5e-4 5e-2 "${seed}"
  run_case MentalArithmetic 4 10 5e-4 1e-2 "${seed}"
done

echo "[$(date -Is)] ALL DONE"
