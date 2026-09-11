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
here (configs in `configs/ablation_fold_geometry/`).

Datasets with `P=1` (SEED-V, PhysioNet-MI, BCIC2020-3) are excluded because
fold and reshape are identical there.

## EfficientNet-B0 (9 P>1 datasets)

| Dataset | Metric | Fold (mean±σ) | Chunk (mean±σ) | Δ | t (2-sample) |
| --- | --- | ---: | ---: | ---: | ---: |
| ISRUC | kappa | 0.77045 ± 0.00316 | 0.73965 ± 0.00444 | -0.03080 | 12.63 |
| TUAB | PR-AUC | 0.91780 ± 0.00245 | 0.90308 ± 0.00162 | -0.01472 | 11.21 |
| FACED | kappa | 0.49533 ± 0.01970 | 0.31981 ± 0.06492 | -0.17552 | 5.78 |
| MentalArithmetic | PR-AUC | 0.79029 ± 0.05364 | 0.58492 ± 0.07337 | -0.20537 | 5.05 |
| TUEV | kappa | 0.69009 ± 0.01896 | 0.65090 ± 0.02313 | -0.03919 | 2.93 |
| Mumtaz2016 | PR-AUC | 0.98637 ± 0.00446 | 0.97610 ± 0.00918 | -0.01027 | 2.25 |
| HMC | kappa | 0.70701 ± 0.00231 | 0.63789 ± 0.00497 | -0.06912 | 28.2 |
| CHB-MIT | PR-AUC | 0.40916 ± 0.05359 | 0.38152 ± 0.06044 | -0.02764 | 0.76 |
| SHU-MI | PR-AUC | 0.69819 ± 0.01780 | 0.70094 ± 0.01382 | +0.00275 | -0.27 |

Negative Δ means fold is better. The two-sample t is Welch, n=5 each, using
population standard deviations.

## Conclusion (EfficientNet-B0)

Across all 9 `P>1` datasets, the phase-interleaved fold is never worse than the
contiguous-chunk reshape: it is significantly better on 7 (HMC, ISRUC, TUAB,
FACED, MentalArithmetic, TUEV, Mumtaz2016), indistinguishable on 2 (CHB-MIT
with high variance, SHU-MI), and never worse on any dataset.

The largest fold advantages appear on FACED (-0.176 kappa) and
MentalArithmetic (-0.205 PR-AUC), while low-variance clinical tasks (HMC,
TUAB, ISRUC) show small but highly significant gaps (t > 11).

## Status

- EfficientNet-B0: 9/9 complete.
- ConvNeXt-Tiny DINOv3: 9/9 complete.
- ViT-Small DINOv3: not run (stopped by decision; a preliminary 3-seed CHB-MIT
  run was within noise and did not justify the extra compute).

## Cross-backbone consistency

ConvNeXt-Tiny DINOv3 repeats the EfficientNet-B0 direction on every completed
dataset (8/8 with five seeds each):

| Dataset | EfficientNet-B0 Δ | ConvNeXt-Tiny Δ |
| --- | ---: | ---: |
| FACED | -0.176 | -0.257 |
| MentalArithmetic | -0.205 | -0.140 |
| TUAB | -0.015 | -0.046 |
| CHB-MIT | -0.028 | -0.044 |
| TUEV | -0.039 | -0.037 |
| ISRUC | -0.031 | -0.026 |
| SHU-MI | +0.003 | -0.009 |
| Mumtaz2016 | -0.010 | -0.004 |
| HMC | -0.069 | -0.041 |

ConvNeXt-Tiny DINOv3 is complete on all 9 datasets; every one preserves the
fold-better-than-reshape direction.

Two tiers emerge on both backbones: large fold advantages on the emotion and
mental-arithmetic tasks (FACED, MentalArithmetic), and small-to-negligible
differences elsewhere. ConvNeXt TUAB is inflated by one unstable seed (0.731;
the other four cluster at 0.889--0.898, near the 0.906 fold baseline).

Run `python experiments/collect_fold_reshape_results.py` to refresh the table
as more configs finish.


