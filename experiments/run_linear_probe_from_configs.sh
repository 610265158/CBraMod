#!/usr/bin/env bash
set -euo pipefail

backbones=(convnext_tiny_dinov3 vit_small_dinov3)
datasets=(BCIC2020-3 CHB-MIT FACED HMC ISRUC MentalArithmetic Mumtaz2016 PhysioNet-MI SEED-V SHU-MI TUAB TUEV)
seeds=(42 43 44 45 46)

for backbone in "${backbones[@]}"; do
  for dataset in "${datasets[@]}"; do
    config="configs/linear_probe/${backbone}/${dataset}.yaml"
    [[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 1; }
    safe_dataset=${dataset,,}; safe_dataset=${safe_dataset//-/_}
    for seed in "${seeds[@]}"; do
      log_dir="experiments/logs/linear_probe_configs/${backbone}/${safe_dataset}/seed${seed}"
      if [[ -d "$log_dir" ]] && rg -q 'Test Evaluation:' "$log_dir"; then
        echo "===== skip completed: ${backbone} ${dataset} seed=${seed} ====="
        continue
      fi
      echo "===== config linear probe: ${backbone} ${dataset} seed=${seed} ====="
      bash experiments/run_downstream.sh --config "$config" --seed "$seed" \
        --model_root "experiments/checkpoints/linear_probe_configs/${backbone}/${safe_dataset}/seed${seed}" \
        --log_root "$log_dir" --device cuda
    done
  done
done
