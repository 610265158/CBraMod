#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="phase_fold_best_3seed_v1"

run_case() {
  local dataset="$1"
  local seed="$2"
  local fold_factor="$3"
  local epochs="$4"
  local lr="$5"
  local weight_decay="$6"
  local metric="$7"
  local run_name="${dataset,,}_p${fold_factor}_seed${seed}"
  run_name="${run_name//-/_}"

  echo "[$(date -Is)] START dataset=${dataset} P=${fold_factor} seed=${seed} epochs=${epochs} lr=${lr} wd=${weight_decay}"
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
    --batch_size 32 \
    --num_workers 4 \
    --lr "${lr}" \
    --weight_decay "${weight_decay}" \
    --multi_lr false \
    --dropout 0.1 \
    --early_stop "${epochs}" \
    --amp true \
    --amp_dtype float16 \
    --test_each_epoch false \
    --run_final_test true \
    --selection_metric "${metric}"
  echo "[$(date -Is)] DONE dataset=${dataset} P=${fold_factor} seed=${seed}"
}

# Seed 3407 is the selected search run: P=1, Kappa/F1=.52500/.62020.
for seed in 3408 3409; do
  run_case BCIC2020-3 "${seed}" 1 30 1e-3 5e-3 kappa
done

# Seed 3407 is the selected search run: P=2, PR/ROC=.97603/.97586.
for seed in 3408 3409; do
  run_case Mumtaz2016 "${seed}" 2 30 5e-4 5e-2 pr_auc
done

echo "[$(date -Is)] ALL DONE"
