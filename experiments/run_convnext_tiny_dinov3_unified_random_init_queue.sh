#!/usr/bin/env bash
# Reproduce the matched random-initialization controls for the unified
# ConvNeXt-Tiny DINOv3 five-seed recipes (num_workers=4, non-persistent).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RUN_ID="${CONVNEXT_RANDOM_INIT_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${CONVNEXT_RANDOM_INIT_QUEUE_ROOT:-experiments/logs/pretrain_vs_random/queue_${RUN_ID}}"
mkdir -p "${QUEUE_ROOT}"

configs=(
  configs/ablation_random_init/convnext_tiny_dinov3_unified/BCIC2020-3.yaml
  configs/ablation_random_init/convnext_tiny_dinov3_unified/CHB-MIT.yaml
  configs/ablation_random_init/convnext_tiny_dinov3_unified/FACED.yaml
  configs/ablation_random_init/convnext_tiny_dinov3_unified/HMC.yaml
  configs/ablation_random_init/convnext_tiny_dinov3_unified/ISRUC.yaml
  configs/ablation_random_init/convnext_tiny_dinov3_unified/MentalArithmetic.yaml
  configs/ablation_random_init/convnext_tiny_dinov3_unified/Mumtaz2016.yaml
  configs/ablation_random_init/convnext_tiny_dinov3_unified/PhysioNet-MI.yaml
  configs/ablation_random_init/convnext_tiny_dinov3_unified/SEED-V.yaml
  configs/ablation_random_init/convnext_tiny_dinov3_unified/SHU-MI.yaml
  configs/ablation_random_init/convnext_tiny_dinov3_unified/TUAB.yaml
  configs/ablation_random_init/convnext_tiny_dinov3_unified/TUEV.yaml
)

failed=()
echo "[$(date -u '+%F %T UTC')] ConvNeXt-Tiny DINOv3 unified random-init queue (w4, non-persistent) started"
echo "Queue root: ${QUEUE_ROOT}"
echo "Configs: ${#configs[@]} (each config uses its YAML protocol seeds)"

for config in "${configs[@]}"; do
  dataset="$(basename "${config}" .yaml)"
  log="${QUEUE_ROOT}/${dataset}.log"
  echo "[$(date -u '+%F %T UTC')] START ${dataset}"
  bash experiments/run_downstream.sh --config "${config}" --cuda 0 --device cuda \
    >"${log}" 2>&1
  status=$?
  if (( status == 0 )); then
    echo "[$(date -u '+%F %T UTC')] DONE  ${dataset}"
  else
    echo "[$(date -u '+%F %T UTC')] FAIL  ${dataset} status=${status}; see ${log}"
    failed+=("${dataset}")
  fi
done

echo "[$(date -u '+%F %T UTC')] ConvNeXt-Tiny DINOv3 unified random-init queue finished"
if (( ${#failed[@]} > 0 )); then
  echo "Failed configs: ${failed[*]}"
  exit 1
fi
echo "All configs completed successfully."
