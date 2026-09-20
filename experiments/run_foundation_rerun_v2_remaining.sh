#!/usr/bin/env bash
# Foundation rerun v2 queue for the seven datasets outside the first sweep:
# CHB-MIT, TUEV, SEED-V, SHU-MI, BCIC2020-3, ISRUC, MentalArithmetic.
#
# Same v2 recipe as the five-dataset sweep: lr 1e-4, warmup 3 epochs at factor
# 0.1, EMA 0.995, weight decay 5e-4, gradient clipping 1.0, early stop 10,
# validation-selected checkpoint and one final test per seed (42-46).
# Per-dataset epochs, batch size and selection metric come from
# configs/downstream.py; input conventions live in configs/foundation.py and
# the resolved per-dataset records under configs/foundation_models/<model>/.
#
# Usage:
#   bash experiments/run_foundation_rerun_v2_remaining.sh
#   MODELS="reve" DATASETS="TUEV" SEEDS="42" bash experiments/run_foundation_rerun_v2_remaining.sh
#
# Failed runs are retried; completed runs are marked with a .done sentinel and
# skipped on re-runs.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS="${MODELS:-cbramod reve}"
DATASETS="${DATASETS:-CHB-MIT TUEV SEED-V SHU-MI BCIC2020-3 ISRUC MentalArithmetic}"
SEEDS="${SEEDS:-42 43 44 45 46}"
LR="${LR:-1e-4}"
RETRIES="${RETRIES:-2}"
WORKERS="${WORKERS:-4}"
RUN_ROOT="${RUN_ROOT:-foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1}"
CBRA_MOD_CHECKPOINT="${CBRA_MOD_CHECKPOINT:-experiments/.third_party/cbramod/pretrained_weights.pth}"
# REVE's 22-layer encoder cannot fit ISRUC's 20-chunk 4D batches at the
# configured batch size 16 (CUDA OOM at ~23.5 GiB); REVE ISRUC therefore runs
# at batch 8.  This is the only per-run batch-size deviation.
REVE_ISRUC_BATCH="${REVE_ISRUC_BATCH:-8}"
# Optional budget overrides for a single (model, dataset) tuning run; empty
# values keep the per-dataset epochs and the v2 early-stop of 10.
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
# Optional REVE readout override ('' keeps the per-dataset spec, 'no' flattens
# all tokens with the context token, 'last' uses the pooled context token).
REVE_POOLING="${REVE_POOLING:-}"
EARLY_STOP_OVERRIDE="${EARLY_STOP_OVERRIDE:-}"
# LMDB-backed datasets run with num_workers=0: fork-shared LMDB handles are a
# known crash source in this pipeline (Pin memory thread / Memo value errors).
LMDB_DATASETS="${LMDB_DATASETS:-SEED-V SHU-MI BCIC2020-3 MentalArithmetic}"

for model in $MODELS; do
  for dataset in $DATASETS; do
    num_workers="$WORKERS"
    for lmdb_dataset in $LMDB_DATASETS; do
      if [ "$dataset" = "$lmdb_dataset" ]; then
        num_workers=0
      fi
    done

    for seed in $SEEDS; do
      safe_dataset="$(printf '%s' "$dataset" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
      model_root="experiments/checkpoints/$RUN_ROOT/$model/seed$seed"
      log_root="experiments/logs/$RUN_ROOT/$model/seed$seed"
      done_file="$model_root/$safe_dataset/.done"

      if [ -f "$done_file" ]; then
        echo "=== skip (already done): $model / $dataset / seed $seed ==="
        continue
      fi

      foundation_args=()
      if [ "$model" = "cbramod" ]; then
        foundation_args=(--foundation_dir "$CBRA_MOD_CHECKPOINT")
      fi

      batch_args=()
      if [ "$model" = "reve" ] && [ "$dataset" = "ISRUC" ]; then
        batch_args=(--batch_size "$REVE_ISRUC_BATCH")
      fi

      pooling_args=()
      if [ -n "$REVE_POOLING" ] && [ "$model" = "reve" ]; then
        pooling_args=(--reve_pooling "$REVE_POOLING")
      fi

      budget_args=()
      if [ -n "$EPOCHS_OVERRIDE" ]; then
        budget_args+=(--epochs "$EPOCHS_OVERRIDE")
      fi
      if [ -n "$EARLY_STOP_OVERRIDE" ]; then
        budget_args+=(--early_stop "$EARLY_STOP_OVERRIDE")
      fi

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
            --warmup_epochs 3 \
            --warmup_start_factor 0.1 \
            --ema_decay 0.995 \
            --weight_decay 0.0005 \
            --clip_value 1.0 \
            --early_stop 10 \
            --test_each_epoch false \
            --run_final_test true \
            --model_root "$model_root" \
            --log_root "$log_root" \
            "${foundation_args[@]}" \
            "${batch_args[@]}" \
            "${pooling_args[@]}" \
            "${budget_args[@]}"; then
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

echo "=== foundation v2 remaining queue finished ($(date '+%F %T')) ==="
