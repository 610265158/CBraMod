#!/usr/bin/env bash
# Matched-pipeline foundation rerun queue.
#
# CBraMod and REVE train with THIS repository's recipe: the per-dataset
# training block in configs/downstream.py (batch size, epochs, weight decay,
# dropout, AMP, label smoothing, EMA, selection metric, early stop) plus the
# locked protocol (validation-selected checkpoint, one final test per seed,
# seeds 42-46).  The only per-model adaptation is the released fine-tuning
# learning rate: --lr 1e-4 for both models.
#
# Data come from the pre-processed datasets under ../BigDownstream via the
# repository loaders; nothing is re-preprocessed.
#
# Usage:
#   bash experiments/run_foundation_rerun.sh
#   MODELS="reve" DATASETS="FACED" SEEDS="42" bash experiments/run_foundation_rerun.sh
#
# Failed runs are reported and skipped; completed runs are marked with a .done
# sentinel inside their checkpoint directory and are skipped on re-runs.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS="${MODELS:-reve cbramod}"
DATASETS="${DATASETS:-FACED TUAB HMC}"
SEEDS="${SEEDS:-42 43 44 45 46}"
LR="${LR:-1e-4}"
WORKERS="${WORKERS:-4}"
RETRIES="${RETRIES:-2}"
# LMDB-backed datasets run with num_workers=0: fork-shared LMDB handles are a
# known crash source in this pipeline (Pin memory thread / Memo value errors).
LMDB_DATASETS="${LMDB_DATASETS:-FACED SEED-V PhysioNet-MI SHU-MI BCIC2020-3 Mumtaz2016 MentalArithmetic}"

for model in $MODELS; do
  for dataset in $DATASETS; do
    for seed in $SEEDS; do
      safe_dataset="$(printf '%s' "$dataset" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
      model_root="experiments/checkpoints/foundation_rerun/$model/seed$seed"
      log_root="experiments/logs/foundation_rerun/$model/seed$seed"
      done_file="$model_root/$safe_dataset/.done"

      if [ -f "$done_file" ]; then
        echo "=== skip (already done): $model / $dataset / seed $seed ==="
        continue
      fi

      num_workers="$WORKERS"
      for lmdb_dataset in $LMDB_DATASETS; do
        if [ "$dataset" = "$lmdb_dataset" ]; then
          num_workers=0
        fi
      done

      attempt=0
      status=1
      while [ "$attempt" -le "$RETRIES" ]; do
        attempt=$((attempt + 1))
        echo "=== start: $model / $dataset / seed $seed (attempt $attempt, workers $num_workers) ($(date '+%F %T')) ==="
        if bash experiments/run_downstream.sh \
            --dataset "$dataset" \
            --cuda 0 \
            --model_arch "$model" \
            --seed "$seed" \
            --lr "$LR" \
            --num_workers "$num_workers" \
            --test_each_epoch false \
            --run_final_test true \
            --model_root "$model_root" \
            --log_root "$log_root"; then
          status=0
          break
        fi
        echo "=== attempt $attempt failed: $model / $dataset / seed $seed ($(date '+%F %T')) ==="
        sleep 30
      done

      if [ "$status" -eq 0 ]; then
        mkdir -p "$(dirname "$done_file")"
        touch "$done_file"
        echo "=== done: $model / $dataset / seed $seed ($(date '+%F %T')) ==="
      else
        echo "=== FAILED after $attempt attempt(s): $model / $dataset / seed $seed ($(date '+%F %T')) ==="
      fi
    done
  done
done

echo "=== foundation rerun queue finished ($(date '+%F %T')) ==="
