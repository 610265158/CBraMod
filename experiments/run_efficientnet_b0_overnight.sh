#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

RUN_ID="${EFFICIENTNET_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${EFFICIENTNET_RUN_ROOT:-experiments/logs/efficientnet_b0/overnight_${RUN_ID}}"
mkdir -p "${RUN_ROOT}"
# The caller redirects stdout/stderr to queue.log for robust nohup operation.

configs=(
  configs/backbones/efficientnet_b0/BCIC2020-3.yaml
  configs/backbones/efficientnet_b0/CHB-MIT.yaml
  configs/backbones/efficientnet_b0/FACED.yaml
  configs/backbones/efficientnet_b0/ISRUC.yaml
  configs/backbones/efficientnet_b0/MentalArithmetic.yaml
  configs/backbones/efficientnet_b0/PhysioNet-MI.yaml
  configs/backbones/efficientnet_b0/SHU-MI.yaml
  configs/backbones/efficientnet_b0/TUEV.yaml
)
seeds=(42 43 44 45 46)
max_oom_retries=5
failed=()

echo "[$(date -u '+%F %T UTC')] EfficientNet-B0 overnight queue started"
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
      "${cmd[@]}" 2>&1 | tee "${log}"
      status=${PIPESTATUS[0]}

      if (( status == 0 )); then
        echo "[$(date -u '+%F %T UTC')] DONE  ${dataset} seed=${seed} batch_size=${batch}"
        completed=1
        break
      fi

      oom=0
      if (( status == 137 || status == 9 )); then
        oom=1
      elif grep -Eqi 'out of memory|cuda out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|not enough memory' "${log}"; then
        oom=1
      fi
      if (( oom == 1 )); then
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

echo "[$(date -u '+%F %T UTC')] EfficientNet-B0 overnight queue finished"
if (( ${#failed[@]} > 0 )); then
  echo "Failed runs:"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo "All queued runs completed successfully."
