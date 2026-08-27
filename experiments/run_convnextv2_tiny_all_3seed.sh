#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
EXPERIMENT_NAME="convnextv2_tiny_lr1e4_3seed_v1"
BACKBONE="convnextv2_tiny"
LR="1e-4"

run_case() {
  local dataset="$1" fold="$2" epochs="$3" batch="$4" wd="$5" metric="$6" mirror="$7"
  for seed in 3407 3408 3409; do
    local name="${dataset,,}_p${fold}_seed${seed}"
    name="${name//-/_}"
    echo "[$(date -Is)] START ${dataset} P=${fold} seed=${seed}"
    bash experiments/run_downstream.sh \
      --dataset "${dataset}" \
      --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${name}" \
      --log_root "experiments/logs/${EXPERIMENT_NAME}/${name}" \
      --device cuda --cuda 0 --seed "${seed}" \
      --vision_fold_factor "${fold}" \
      --backbone_name "${BACKBONE}" \
      --use_pretrained_weights true \
      --epochs "${epochs}" --batch_size "${batch}" --num_workers 4 \
      --lr "${LR}" --weight_decay "${wd}" --clip_value -1 --multi_lr false \
      --dropout 0.1 --early_stop "${epochs}" \
      --mirror_augmentation "${mirror}" --mirror_prob 0.5 \
      --time_roll_augmentation false --amplitude_scale_augmentation false \
      --amp true --amp_dtype float16 \
      --test_each_epoch false --run_final_test true --selection_metric "${metric}" \
      || echo "[$(date -Is)] FAILED ${dataset} P=${fold} seed=${seed} (continuing)"
    echo "[$(date -Is)] DONE ${dataset} P=${fold} seed=${seed}"
  done
}

run_case BCIC2020-3 1 30 32 5e-3 kappa false
run_case Mumtaz2016 2 30 32 5e-2 pr_auc false
run_case MentalArithmetic 2 10 32 1e-2 pr_auc false
run_case TUEV 4 10 32 5e-3 kappa false
run_case ISRUC 8 50 16 5e-3 kappa true
run_case SEED-V 8 50 32 5e-3 kappa false
run_case CHB-MIT 2 10 32 5e-3 pr_auc false
run_case TUAB 2 5 32 5e-4 pr_auc false
run_case FACED 2 50 32 5e-3 kappa false
run_case PhysioNet-MI 1 30 32 5e-3 kappa false
run_case SHU-MI 4 20 32 5e-3 pr_auc false

echo "[$(date -Is)] ALL DONE"
