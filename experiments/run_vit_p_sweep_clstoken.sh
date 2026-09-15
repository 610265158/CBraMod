#!/usr/bin/env bash
# ViT-Small DINOv3 fold-factor (P) sweep -- corrected CLS-token recipe rerun.
#
# Scope: CHB-MIT, TUEV, MentalArithmetic x P in {1, 2, 8} x seeds 42-46
# (9 configs, 45 runs). The P=4 rows are intentionally not queued: they
# coincide with the finalized main recipes, and their results are sourced from
# configs/backbones/vit_small_dinov3/<dataset>.yaml.
#
# Launch: CUDA_ID=0 bash experiments/run_vit_p_sweep_clstoken.sh
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CUDA_ID="${CUDA_ID:-0}"

failed=0
run_one() {
  cfg="$1"
  echo "===== START ${cfg} $(date +%H:%M:%S) ====="
  if bash experiments/run_downstream.sh --config "${cfg}" --device cuda --cuda "${CUDA_ID}"; then
    echo "===== DONE ${cfg} (ok) $(date +%H:%M:%S) ====="
  else
    failed=$((failed + 1))
    echo "===== FAILED ${cfg} $(date +%H:%M:%S) ====="
  fi
}

run_one configs/ablation_p/vit_small_dinov3/CHB-MIT_P1.yaml
run_one configs/ablation_p/vit_small_dinov3/CHB-MIT_P2.yaml
run_one configs/ablation_p/vit_small_dinov3/CHB-MIT_P8.yaml
run_one configs/ablation_p/vit_small_dinov3/TUEV_P1.yaml
run_one configs/ablation_p/vit_small_dinov3/TUEV_P2.yaml
run_one configs/ablation_p/vit_small_dinov3/TUEV_P8.yaml
run_one configs/ablation_p/vit_small_dinov3/MentalArithmetic_P1.yaml
run_one configs/ablation_p/vit_small_dinov3/MentalArithmetic_P2.yaml
run_one configs/ablation_p/vit_small_dinov3/MentalArithmetic_P8.yaml

echo "===== ALL DONE $(date +%H:%M:%S) failures=${failed} ====="
exit $((failed > 0))
