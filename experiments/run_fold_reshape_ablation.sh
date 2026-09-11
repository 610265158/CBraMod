#!/usr/bin/env bash
set -u
run_one() {
  cfg="$1"
  echo "===== START $cfg $(date +%H:%M:%S) ====="
  python experiments/downstream_11.py --config "$cfg" --cuda 0 --device cuda
  echo "===== DONE $cfg (exit $?) $(date +%H:%M:%S) ====="
}

# ---- efficientnet_b0 ----
run_one configs/ablation_fold_geometry/efficientnet_b0/CHB-MIT_chunk.yaml
run_one configs/ablation_fold_geometry/efficientnet_b0/TUAB_chunk.yaml
run_one configs/ablation_fold_geometry/efficientnet_b0/TUEV_chunk.yaml
run_one configs/ablation_fold_geometry/efficientnet_b0/ISRUC_chunk.yaml
run_one configs/ablation_fold_geometry/efficientnet_b0/FACED_chunk.yaml
run_one configs/ablation_fold_geometry/efficientnet_b0/SHU-MI_chunk.yaml
run_one configs/ablation_fold_geometry/efficientnet_b0/Mumtaz2016_chunk.yaml
run_one configs/ablation_fold_geometry/efficientnet_b0/MentalArithmetic_chunk.yaml
run_one configs/ablation_fold_geometry/efficientnet_b0/HMC_chunk.yaml
# ---- convnext_tiny_dinov3 ----
run_one configs/ablation_fold_geometry/convnext_tiny_dinov3/CHB-MIT_chunk.yaml
run_one configs/ablation_fold_geometry/convnext_tiny_dinov3/TUAB_chunk.yaml
run_one configs/ablation_fold_geometry/convnext_tiny_dinov3/TUEV_chunk.yaml
run_one configs/ablation_fold_geometry/convnext_tiny_dinov3/ISRUC_chunk.yaml
run_one configs/ablation_fold_geometry/convnext_tiny_dinov3/FACED_chunk.yaml
run_one configs/ablation_fold_geometry/convnext_tiny_dinov3/SHU-MI_chunk.yaml
run_one configs/ablation_fold_geometry/convnext_tiny_dinov3/Mumtaz2016_chunk.yaml
run_one configs/ablation_fold_geometry/convnext_tiny_dinov3/MentalArithmetic_chunk.yaml
run_one configs/ablation_fold_geometry/convnext_tiny_dinov3/HMC_chunk.yaml
# ---- vit_small_dinov3 ----
run_one configs/ablation_fold_geometry/vit_small_dinov3/CHB-MIT_chunk.yaml
run_one configs/ablation_fold_geometry/vit_small_dinov3/TUAB_chunk.yaml
run_one configs/ablation_fold_geometry/vit_small_dinov3/TUEV_chunk.yaml
run_one configs/ablation_fold_geometry/vit_small_dinov3/ISRUC_chunk.yaml
run_one configs/ablation_fold_geometry/vit_small_dinov3/FACED_chunk.yaml
run_one configs/ablation_fold_geometry/vit_small_dinov3/SHU-MI_chunk.yaml
run_one configs/ablation_fold_geometry/vit_small_dinov3/Mumtaz2016_chunk.yaml
run_one configs/ablation_fold_geometry/vit_small_dinov3/MentalArithmetic_chunk.yaml
run_one configs/ablation_fold_geometry/vit_small_dinov3/HMC_chunk.yaml

echo "===== ALL DONE $(date +%H:%M:%S) ====="
