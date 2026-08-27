#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PILOT_EXPERIMENT="phase_fold_remaining5_pilot_v1"
PILOT_LAUNCHER="experiments/logs/${PILOT_EXPERIMENT}/launcher.log"
EXPERIMENT_NAME="phase_fold_remaining5_3seed_v1"

run_case() {
  local dataset="$1"
  local fold_factor="$2"
  local epochs="$3"
  local batch_size="$4"
  local lr="$5"
  local weight_decay="$6"
  local metric="$7"
  local mirror="$8"
  local clip_value="$9"
  local seed="${10}"
  local run_name="${dataset,,}_p${fold_factor}_seed${seed}"
  run_name="${run_name//-/_}"

  echo "[$(date -Is)] START dataset=${dataset} P=${fold_factor} seed=${seed} epochs=${epochs} batch=${batch_size} lr=${lr} wd=${weight_decay} mirror=${mirror}"
  bash experiments/run_downstream.sh \
    --dataset "${dataset}" \
    --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${run_name}" \
    --log_root "experiments/logs/${EXPERIMENT_NAME}/${run_name}" \
    --device cuda \
    --cuda 0 \
    --seed "${seed}" \
    --vision_fold_factor "${fold_factor}" \
    --backbone_name efficientnet_b0 \
    --use_pretrained_weights true \
    --epochs "${epochs}" \
    --batch_size "${batch_size}" \
    --num_workers 4 \
    --lr "${lr}" \
    --weight_decay "${weight_decay}" \
    --clip_value "${clip_value}" \
    --multi_lr false \
    --dropout 0.1 \
    --early_stop "${epochs}" \
    --mirror_augmentation "${mirror}" \
    --mirror_prob 0.5 \
    --amplitude_scale_augmentation false \
    --amp true \
    --amp_dtype float16 \
    --test_each_epoch false \
    --run_final_test true \
    --selection_metric "${metric}"
  echo "[$(date -Is)] DONE dataset=${dataset} P=${fold_factor} seed=${seed}"
}

echo "[$(date -Is)] WAIT for ${PILOT_EXPERIMENT}"
until [[ -f "${PILOT_LAUNCHER}" ]] && grep -q '] ALL DONE' "${PILOT_LAUNCHER}"; do
  sleep 30
done
echo "[$(date -Is)] PILOT DONE; starting seeds 3408 and 3409"

for seed in 3408 3409; do
  run_case TUEV 2 50 32 3e-4 5e-3 kappa false -1 "${seed}"
  run_case ISRUC 8 50 16 1e-3 5e-3 kappa true -1 "${seed}"
  run_case SEED-V 8 50 32 5e-4 5e-3 kappa false -1 "${seed}"
  run_case CHB-MIT 2 10 32 1e-3 5e-3 pr_auc false -1 "${seed}"
  run_case TUAB 2 5 32 1e-3 5e-4 pr_auc false 1 "${seed}"
done

echo "[$(date -Is)] ALL DONE"
