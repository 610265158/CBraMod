#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RUN_ID="${CONVNEXT_DINOV3_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${CONVNEXT_DINOV3_QUEUE_ROOT:-experiments/logs/convnext_tiny_dinov3/queue_${RUN_ID}}"
mkdir -p "${QUEUE_ROOT}"

configs=(
  configs/backbones/convnext_tiny_dinov3/BCIC2020-3.yaml
  configs/backbones/convnext_tiny_dinov3/FACED.yaml
  configs/backbones/convnext_tiny_dinov3/ISRUC.yaml
  configs/backbones/convnext_tiny_dinov3/Mumtaz2016.yaml
  configs/backbones/convnext_tiny_dinov3/PhysioNet-MI.yaml
  configs/backbones/convnext_tiny_dinov3/SHU-MI.yaml
  configs/backbones/convnext_tiny_dinov3/TUEV.yaml
)

failed=()
echo "[$(date -u '+%F %T UTC')] ConvNeXt-Tiny DINOv3 queue started"
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

echo "[$(date -u '+%F %T UTC')] ConvNeXt-Tiny DINOv3 queue finished"
if (( ${#failed[@]} > 0 )); then
  echo "Failed configs: ${failed[*]}"
  exit 1
fi
echo "All configs completed successfully."
