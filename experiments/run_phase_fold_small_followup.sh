#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="phase_fold_small_followup_v1"

run_case() {
  local dataset="$1"
  local fold_factor="$2"
  local seed="$3"
  local epochs="$4"
  local lr="$5"
  local weight_decay="$6"
  local multi_lr="$7"
  local backbone_lr_scale="$8"
  local metric="$9"
  local tag="${10}"
  local run_name="${dataset,,}_${tag}_p${fold_factor}_seed${seed}"
  run_name="${run_name//-/_}"

  echo "[$(date -Is)] START dataset=${dataset} tag=${tag} P=${fold_factor} seed=${seed} epochs=${epochs} lr=${lr} wd=${weight_decay} multi_lr=${multi_lr}"
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
    --multi_lr "${multi_lr}" \
    --backbone_lr_scale "${backbone_lr_scale}" \
    --dropout 0.1 \
    --early_stop "${epochs}" \
    --amp true \
    --amp_dtype float16 \
    --test_each_epoch false \
    --run_final_test true \
    --selection_metric "${metric}"
  echo "[$(date -Is)] DONE dataset=${dataset} tag=${tag} P=${fold_factor} seed=${seed}"
}

# BCIC2020-3: compare no folding and P=2, then test discriminative LR.
run_case BCIC2020-3 1 3407 30 3e-4 1e-2 false 0.1 kappa full_lr3e4
run_case BCIC2020-3 2 3407 30 3e-4 1e-2 false 0.1 kappa full_lr3e4
run_case BCIC2020-3 1 3407 30 1e-3 5e-3 false 0.1 kappa full_lr1e3
run_case BCIC2020-3 2 3407 30 1e-3 5e-3 false 0.1 kappa full_lr1e3
run_case BCIC2020-3 1 3407 30 1e-3 1e-2 true 0.1 kappa multi_lr1e3
run_case BCIC2020-3 2 3407 30 1e-3 1e-2 true 0.1 kappa multi_lr1e3

# Mumtaz2016: pilot converged quickly; test the longer, stronger-WD schedule.
run_case Mumtaz2016 2 3407 30 5e-4 5e-2 false 0.1 pr_auc wd5e2
run_case Mumtaz2016 4 3407 30 5e-4 5e-2 false 0.1 pr_auc wd5e2

# MentalArithmetic: exact pilot configuration, completing the three-seed set.
run_case MentalArithmetic 2 3408 10 5e-4 1e-2 false 0.1 pr_auc pilot_repro
run_case MentalArithmetic 2 3409 10 5e-4 1e-2 false 0.1 pr_auc pilot_repro

echo "[$(date -Is)] ALL DONE"
