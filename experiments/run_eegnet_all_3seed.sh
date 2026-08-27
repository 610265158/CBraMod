#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="eegnet_8_2_all_3seed_v1"

run_case() {
  local dataset="$1" epochs="$2" batch="$3" metric="$4" mirror="$5"
  for seed in 3407 3408 3409; do
    local name="${dataset,,}_seed${seed}"
    name="${name//-/_}"
    echo "[$(date -Is)] START EEGNet dataset=${dataset} seed=${seed}"
    bash experiments/run_downstream.sh \
      --dataset "${dataset}" \
      --model_arch eegnet \
      --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${name}" \
      --log_root "experiments/logs/${EXPERIMENT_NAME}/${name}" \
      --device cuda --cuda 0 --seed "${seed}" \
      --epochs "${epochs}" --batch_size "${batch}" --num_workers 4 \
      --lr 1e-3 --optimizer Adam --weight_decay 0 \
      --label_smoothing 0 --dropout 0.5 --clip_value -1 \
      --multi_lr false --use_pretrained_weights false \
      --mirror_augmentation "${mirror}" --mirror_prob 0.5 \
      --time_roll_augmentation false --amplitude_scale_augmentation false \
      --amp true --amp_dtype float16 \
      --early_stop "${epochs}" \
      --test_each_epoch false --run_final_test true \
      --selection_metric "${metric}"
    echo "[$(date -Is)] DONE EEGNet dataset=${dataset} seed=${seed}"
  done
}

run_case CHB-MIT 10 32 pr_auc false
run_case TUAB 5 32 pr_auc false
run_case TUEV 10 32 kappa false
run_case ISRUC 50 16 kappa true
run_case FACED 50 32 kappa false
run_case SEED-V 50 32 kappa false
run_case PhysioNet-MI 30 32 kappa false
run_case SHU-MI 20 32 pr_auc false
run_case BCIC2020-3 30 32 kappa false
run_case Mumtaz2016 30 32 pr_auc false
run_case MentalArithmetic 10 32 pr_auc false

echo "[$(date -Is)] ALL DONE"
