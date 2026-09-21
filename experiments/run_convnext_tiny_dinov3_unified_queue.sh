#!/usr/bin/env bash
# Reproduce the locked unified ConvNeXt-Tiny DINOv3 five-seed recipe with
# num_workers=4 and non-persistent DataLoader workers.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RUN_ID="${CONVNEXT_DINOV3_UNIFIED_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${CONVNEXT_DINOV3_UNIFIED_QUEUE_ROOT:-experiments/logs/convnext_tiny_dinov3_unified/queue_${RUN_ID}}"
mkdir -p "${QUEUE_ROOT}"

configs=(
  configs/backbones/convnext_tiny_dinov3_unified/BCIC2020-3.yaml
  configs/backbones/convnext_tiny_dinov3_unified/CHB-MIT.yaml
  configs/backbones/convnext_tiny_dinov3_unified/FACED.yaml
  configs/backbones/convnext_tiny_dinov3_unified/HMC.yaml
  configs/backbones/convnext_tiny_dinov3_unified/ISRUC.yaml
  configs/backbones/convnext_tiny_dinov3_unified/MentalArithmetic.yaml
  configs/backbones/convnext_tiny_dinov3_unified/Mumtaz2016.yaml
  configs/backbones/convnext_tiny_dinov3_unified/PhysioNet-MI.yaml
  configs/backbones/convnext_tiny_dinov3_unified/SEED-V.yaml
  configs/backbones/convnext_tiny_dinov3_unified/SHU-MI.yaml
  configs/backbones/convnext_tiny_dinov3_unified/TUAB.yaml
  configs/backbones/convnext_tiny_dinov3_unified/TUEV.yaml
)

failed=()
echo "[$(date -u '+%F %T UTC')] ConvNeXt-Tiny DINOv3 unified queue (w4, non-persistent) started"
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

echo "[$(date -u '+%F %T UTC')] ConvNeXt-Tiny DINOv3 unified queue finished"
if (( ${#failed[@]} > 0 )); then
  echo "Failed configs: ${failed[*]}"
  exit 1
fi
echo "All configs completed successfully."
