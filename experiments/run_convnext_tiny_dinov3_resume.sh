#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
RUN_ID="${CONVNEXT_DINOV3_RESUME_ID:-$(date -u +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${CONVNEXT_DINOV3_RESUME_ROOT:-experiments/logs/convnext_tiny_dinov3/resume_${RUN_ID}}"
mkdir -p "${QUEUE_ROOT}"

tasks=(
  "FACED:44" "FACED:45" "FACED:46"
  "ISRUC:42" "ISRUC:43" "ISRUC:44" "ISRUC:45" "ISRUC:46"
  "Mumtaz2016:42" "Mumtaz2016:43" "Mumtaz2016:44" "Mumtaz2016:45" "Mumtaz2016:46"
  "PhysioNet-MI:42" "PhysioNet-MI:43" "PhysioNet-MI:44" "PhysioNet-MI:45" "PhysioNet-MI:46"
  "SHU-MI:42" "SHU-MI:43" "SHU-MI:44" "SHU-MI:45" "SHU-MI:46"
  "TUEV:42" "TUEV:43" "TUEV:44" "TUEV:45" "TUEV:46"
)

failed=()
echo "[$(date -u '+%F %T UTC')] Resume queue started; root=${QUEUE_ROOT}"
for task in "${tasks[@]}"; do
  dataset="${task%%:*}"
  seed="${task##*:}"
  config="configs/backbones/convnext_tiny_dinov3/${dataset}.yaml"
  safe="${dataset//-/_}"
  log="${QUEUE_ROOT}/${safe}_seed${seed}.log"
  echo "[$(date -u '+%F %T UTC')] START ${dataset} seed=${seed}"
  bash experiments/run_downstream.sh --config "${config}" --seed "${seed}" --cuda 0 --device cuda \
    >"${log}" 2>&1
  status=$?
  if (( status == 0 )); then
    echo "[$(date -u '+%F %T UTC')] DONE  ${dataset} seed=${seed}"
  else
    echo "[$(date -u '+%F %T UTC')] FAIL  ${dataset} seed=${seed} status=${status}; see ${log}"
    failed+=("${dataset}:seed${seed}")
  fi
done
echo "[$(date -u '+%F %T UTC')] Resume queue finished"
if (( ${#failed[@]} > 0 )); then
  echo "Failed tasks: ${failed[*]}"
  exit 1
fi
echo "All resume tasks completed successfully."
