#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CUDA_ID="${CUDA_ID:-0}"

# Random-init half of the pretrain-vs-random ablation for the newly added
# datasets. Each config is the locked recipe with use_pretrained_weights=false.
# The pretrained baselines are already recorded in configs/backbones/*/.
CONFIGS=(
  configs/ablation_random_init/ISRUC_efficientnet_b0.yaml
  configs/ablation_random_init/ISRUC_convnext_tiny_dinov3.yaml
  configs/ablation_random_init/ISRUC_vit_small_dinov3.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "[pretrain_vs_random] $(basename "$cfg") start: $(date)"
  bash experiments/run_downstream.sh --config "$cfg" --device cuda --cuda "${CUDA_ID}"
  echo "[pretrain_vs_random] $(basename "$cfg") done: $(date)"
done

echo "[pretrain_vs_random] ALL DONE: $(date)"
