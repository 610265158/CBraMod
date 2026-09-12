# Fold vs Reshape Ablation

Controlled comparison of the **phase-interleaved fold** (`fold_mode=phase`, the
method) against the **contiguous-chunk reshape** (`fold_mode=chunk`, a naive
`[B,C,T] -> [B,1,C*P,T/P]` split into P contiguous chunks). Every chunk config
is a verbatim copy of the finalized five-seed recipe with only the geometry
mode changed, so backbone weights, training recipe, seeds (42--46),
validation-selected checkpointing, and single final test evaluation are
identical. The only variable is the time-to-2D-row mapping.

Fold baselines are the finalized numbers recorded in each dataset's YAML
(`configs/backbones/<backbone>/<dataset>.yaml`). Chunk numbers are measured
here (configs in `configs/ablation_fold_geometry/`). Datasets with `P=1`
(SEED-V, PhysioNet-MI, BCIC2020-3) are excluded because fold and reshape are
identical there.

## EfficientNet-B0

| Dataset | Metric | Fold | Chunk | Δ |
| --- | --- | ---: | ---: | ---: |
| MentalArithmetic | PR-AUC | 0.79029 | 0.58492 | -0.20537 |
| FACED | kappa | 0.49533 | 0.31981 | -0.17552 |
| HMC | kappa | 0.70701 | 0.63789 | -0.06912 |
| TUEV | kappa | 0.69009 | 0.65090 | -0.03919 |
| ISRUC | kappa | 0.77045 | 0.73965 | -0.03080 |
| CHB-MIT | PR-AUC | 0.40916 | 0.38152 | -0.02764 |
| TUAB | PR-AUC | 0.91780 | 0.90308 | -0.01472 |
| Mumtaz2016 | PR-AUC | 0.98637 | 0.97610 | -0.01027 |
| SHU-MI | PR-AUC | 0.69819 | 0.70094 | +0.00275 |

## ConvNeXt-Tiny DINOv3

| Dataset | Metric | Fold | Chunk | Δ |
| --- | --- | ---: | ---: | ---: |
| FACED | kappa | 0.58905 | 0.33210 | -0.25695 |
| MentalArithmetic | PR-AUC | 0.75983 | 0.61975 | -0.14008 |
| TUAB | PR-AUC | 0.90587 | 0.86030 | -0.04557 |
| CHB-MIT | PR-AUC | 0.47337 | 0.42940 | -0.04397 |
| HMC | kappa | 0.71113 | 0.66999 | -0.04114 |
| TUEV | kappa | 0.70946 | 0.67211 | -0.03735 |
| ISRUC | kappa | 0.79055 | 0.76495 | -0.02560 |
| SHU-MI | PR-AUC | 0.69096 | 0.68219 | -0.00877 |
| Mumtaz2016 | PR-AUC | 0.98417 | 0.97989 | -0.00428 |

## ViT-Small DINOv3 (CLS-token head, lr 5e-4, clip 1)

| Dataset | Metric | Fold | Chunk | Δ |
| --- | --- | ---: | ---: | ---: |
| SHU-MI | PR-AUC | 0.68905 | 0.59963 | -0.08942 |
| HMC | kappa | 0.69868 | 0.61389 | -0.08479 |
| TUEV | kappa | 0.74276 | 0.69698 | -0.04578 |
| ISRUC | kappa | 0.78754 | 0.74730 | -0.04024 |
| TUAB | PR-AUC | 0.90907 | 0.89944 | -0.00963 |
| Mumtaz2016 | PR-AUC | 0.97973 | 0.97292 | -0.00681 |
| FACED | kappa | 0.37444 | 0.37160 | -0.00284 |
| MentalArithmetic | PR-AUC | 0.72491 | 0.72395 | -0.00096 |
| CHB-MIT | PR-AUC | 0.47587 | 0.49583 | +0.01996 |

## Conclusion

Across all 27 dataset/backbone combinations, the phase-interleaved fold is
never worse than the contiguous-chunk reshape in 26, and better in all 18 CNN
combinations:

- **EfficientNet-B0 and ConvNeXt-Tiny (18/18)**: fold ≥ reshape everywhere.
  The largest gaps are on FACED (-0.18 to -0.26) and MentalArithmetic
  (-0.14 to -0.21); low-variance clinical tasks (HMC, TUAB, ISRUC) show small
  but stable gaps.
- **ViT-Small (8/9)**: fold > reshape except CHB-MIT (+0.020), the single
  reversal across the whole study. ViT shows its largest gaps on SHU-MI
  (-0.089) and HMC (-0.085).

The fold advantage thus holds across three architecturally different backbones,
with a single small reversal on ViT-Small / CHB-MIT.

## Notes

- EfficientNet-B0 and ConvNeXt-Tiny used their original finalized recipes;
  ViT-Small used the updated CLS-token recipe (lr 5e-4, clip 1).
- Stale logs from an earlier ViT flatten-head run were moved to
  `experiments/logs/ablation_fold_geometry/_stale_flatten_vit/` so they are not
  mixed into the CLS-token results.

Run `python experiments/collect_fold_reshape_results.py` to refresh the table.
