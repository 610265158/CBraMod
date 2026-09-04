#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

RUN_ID="${CONVNEXT_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${CONVNEXT_RUN_ROOT:-experiments/logs/convnextv2_tiny_384/overnight_${RUN_ID}}"
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/queue.log") 2>&1

configs=(
  configs/backbones/convnextv2_tiny_384/BCIC2020-3.yaml
  configs/backbones/convnextv2_tiny_384/CHB-MIT.yaml
  configs/backbones/convnextv2_tiny_384/FACED.yaml
  configs/backbones/convnextv2_tiny_384/ISRUC.yaml
  configs/backbones/convnextv2_tiny_384/MentalArithmetic.yaml
  configs/backbones/convnextv2_tiny_384/PhysioNet-MI.yaml
  configs/backbones/convnextv2_tiny_384/SHU-MI.yaml
  configs/backbones/convnextv2_tiny_384/TUEV.yaml
)
seeds=(42 43 44 45 46)
max_oom_retries=5
failed=()

echo "[$(date -u '+%F %T UTC')] ConvNeXtV2-Tiny 384 overnight queue started"
echo "Run root: ${RUN_ROOT}"
echo "Tasks: ${#configs[@]} configs x ${#seeds[@]} seeds; TUAB excluded"

for config in "${configs[@]}"; do
  dataset="$(basename "${config}" .yaml)"
  initial_batch="$(awk '/^  batch_size:/ {print $2; exit}' "${config}")"
  if [[ -z "${initial_batch}" || ! "${initial_batch}" =~ ^[0-9]+$ ]]; then
    echo "[${dataset}] could not read batch_size from ${config}; skipping config"
    failed+=("${dataset} all-seeds(batch-size-read)")
    continue
  fi

  for seed in "${seeds[@]}"; do
    batch="${initial_batch}"
    attempt=1
    completed=0
    while (( attempt <= max_oom_retries + 1 )); do
      log="${RUN_ROOT}/${dataset}_seed${seed}_attempt${attempt}_bs${batch}.log"
      cmd=(bash experiments/run_downstream.sh
        --config "${config}"
        --seed "${seed}"
        --cuda 0
        --device cuda)
      if [[ "${batch}" != "${initial_batch}" ]]; then
        cmd+=(--batch_size "${batch}")
      fi

      echo "[$(date -u '+%F %T UTC')] START ${dataset} seed=${seed} batch_size=${batch} attempt=${attempt}"
      set +e
      "${cmd[@]}" 2>&1 | tee "${log}"
      status=${PIPESTATUS[0]}
      set -e

      if (( status == 0 )); then
        echo "[$(date -u '+%F %T UTC')] DONE  ${dataset} seed=${seed} batch_size=${batch}"
        completed=1
        break
      fi

      if grep -Eqi 'out of memory|cuda out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|not enough memory' "${log}"; then
        next_batch=$(( batch / 2 ))
        if (( next_batch >= 1 && next_batch < batch )); then
          echo "[$(date -u '+%F %T UTC')] OOM   ${dataset} seed=${seed}; retrying with batch_size=${next_batch}"
          batch="${next_batch}"
          attempt=$(( attempt + 1 ))
          continue
        fi
      fi

      echo "[$(date -u '+%F %T UTC')] FAIL  ${dataset} seed=${seed} status=${status}; see ${log}"
      failed+=("${dataset} seed${seed}")
      break
    done

    if (( completed == 0 && attempt > max_oom_retries + 1 )); then
      echo "[$(date -u '+%F %T UTC')] FAIL  ${dataset} seed=${seed}: exhausted OOM retries"
      failed+=("${dataset} seed${seed}(oom-retries)")
    fi
  done
done

echo "[$(date -u '+%F %T UTC')] ConvNeXtV2-Tiny 384 overnight queue finished"
if (( ${#failed[@]} > 0 )); then
  echo "Failed runs:"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo "All queued runs completed successfully."
