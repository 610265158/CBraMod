#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="phase_fold_small_pilot_v1"
SEED=3407
FOLD_FACTORS=(2 4)
DATASETS=(MentalArithmetic Mumtaz2016 BCIC2020-3)

selection_metric() {
  case "$1" in
    MentalArithmetic|Mumtaz2016) echo pr_auc ;;
    BCIC2020-3) echo kappa ;;
    *) echo auto ;;
  esac
}

safe_name() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr '-' '_'
}

for dataset in "${DATASETS[@]}"; do
  for fold_factor in "${FOLD_FACTORS[@]}"; do
    run_name="$(safe_name "${dataset}")_p${fold_factor}_seed${SEED}"
    echo "[$(date -Is)] START dataset=${dataset} P=${fold_factor} seed=${SEED}"
    bash experiments/run_downstream.sh \
      --dataset "${dataset}" \
      --model_root "experiments/checkpoints/${EXPERIMENT_NAME}/${run_name}" \
      --log_root "experiments/logs/${EXPERIMENT_NAME}/${run_name}" \
      --device cuda \
      --cuda 0 \
      --seed "${SEED}" \
      --vision_fold_factor "${fold_factor}" \
      --backbone_name efficientnet_b0 \
      --use_pretrained_weights true \
      --epochs 10 \
      --batch_size 32 \
      --num_workers 4 \
      --lr 5e-4 \
      --weight_decay 1e-2 \
      --dropout 0.1 \
      --early_stop 10 \
      --amp true \
      --amp_dtype float16 \
      --test_each_epoch false \
      --run_final_test true \
      --selection_metric "$(selection_metric "${dataset}")"
    echo "[$(date -Is)] DONE dataset=${dataset} P=${fold_factor} seed=${SEED}"
  done
done

echo "[$(date -Is)] ALL DONE"
