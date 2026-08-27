#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="phase_fold_remaining5_pilot_v1"
SEED=3407

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
  local run_name="${dataset,,}_p${fold_factor}_seed${SEED}"
  run_name="${run_name//-/_}"

  echo "[$(date -Is)] START dataset=${dataset} P=${fold_factor} seed=${SEED} epochs=${epochs} batch=${batch_size} lr=${lr} wd=${weight_decay} mirror=${mirror}"
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
  echo "[$(date -Is)] DONE dataset=${dataset} P=${fold_factor} seed=${SEED}"
}

# Hand over the single GPU after the active SHU three-seed queue completes.
while pgrep -f '[r]un_shu_p4_select_and_3seed.sh' >/dev/null; do
  echo "[$(date -Is)] WAIT SHU three-seed queue"
  sleep 15
done

# Medium/large tasks first so useful results arrive before the two largest sets.
run_case TUEV 2 50 32 3e-4 5e-3 kappa false -1
run_case ISRUC 8 50 16 1e-3 5e-3 kappa true -1
run_case SEED-V 8 50 32 5e-4 5e-3 kappa false -1
run_case CHB-MIT 2 10 32 1e-3 5e-3 pr_auc false -1
run_case TUAB 2 5 32 1e-3 5e-4 pr_auc false 1

echo "[$(date -Is)] ALL DONE"
