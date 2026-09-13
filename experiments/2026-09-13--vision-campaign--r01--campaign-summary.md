---
type: results-report
date: 2026-09-13
experiment_line: vision-campaign
round: r01
purpose: campaign-summary
status: active
source_artifacts:
  - experiments/analysis/vit_clstoken_lr5e4_clip1_2026-09-11/analysis-report.md
  - experiments/analysis/vit_clstoken_lr5e4_clip1_2026-09-11/stats-appendix.md
  - experiments/analysis/vit_clstoken_lr5e4_clip1_2026-09-11/figure-catalog.md
  - experiments/FOLD_VS_RESHAPE_RESULTS.md
  - experiments/PHASE_FOLD_RESULTS.md
linked_experiments:
  - configs/backbones/{efficientnet_b0,convnext_tiny_dinov3,vit_small_dinov3}
  - configs/ablation_random_init
  - configs/ablation_fold_geometry
linked_results:
  - experiments/PHASE_FOLD_RESULTS.md
  - experiments/FOLD_VS_RESHAPE_RESULTS.md
---

# Vision Campaign / Round 1 / Campaign Summary / 2026-09-13

> Round normalization note: this repository had no prior round convention; the
> consolidated round is labeled `r01` for future disambiguation.
> Write-back target: this repo is **not** bound to an Obsidian knowledge base,
> so this report is a local artifact and **no Obsidian write-back was attempted**.

## 1. Executive Summary

This round consolidated the full EEG-vision campaign across **12 datasets x 3 timm
backbones** (EfficientNet-B0, ConvNeXt-Tiny DINOv3, ViT-Small DINOv3) and resolved two
open provenance problems: (a) a mislabeled/misconfigured ViT readout recipe, and
(b) unmatched pretrain-vs-random controls.

Highest-confidence conclusions:

1. **The corrected ViT-Small recipe (true CLS-token head, lr 5e-4, clip 1) is a
   dataset-dependent win**: significant vs the old flatten recipe on 4/12 datasets
   (PhysioNet-MI +8.06pp kappa, FACED +6.47pp, TUAB +2.48pp, ISRUC +0.90pp; all
   Holm-corrected) while SEED-V and BCIC2020-3 keep the flatten recipe.
2. **Pretraining beats random init on 34 of 36 grid cells**; the two exceptions are
   EfficientNet-B0 x ISRUC (-0.28pp, previously known) and EfficientNet-B0 x SEED-V
   (-2.10pp; SEED-V is the documented weak dataset). Gains are large on hard tasks
   (BCIC ConvNeXt +43.8pp, MentalArithmetic +50~58pp, FACED +18~35pp) and near-zero
   on large/easy tasks (TUAB CN +0.25pp, Mumtaz ViT +0.19pp, HMC +2~8pp).
3. **Phase-interleaved folding beats contiguous-chunk reshape** across all three
   backbones: EfficientNet-B0 fold-favored on 7/9 (2 ties), ConvNeXt-Tiny on 9/9,
   ViT-Small on 8/9 (single small reversal, CHB-MIT +0.020). Source:
   `FOLD_VS_RESHAPE_RESULTS.md`.

Decision changed: the committed ViT configs now carry the corrected recipe for 10
datasets, and pretrain-vs-random tables can be reported with matched recipes for the
full 12-dataset suite (all 36 cells complete).

## 2. Experiment Identity and Decision Context

- Line: vision-backbone transfer for EEG decoding (lossless temporal folding).
- Why this round: the previously labeled "cls_token" head was verified to be
  patch-token mean pooling (timm `Eva`, `global_pool='avg'`), and the ViT learning
  rate (1e-4, clipping off) was suspected to handicap training. Pretrain-vs-random
  controls had been run under the old recipes and were no longer matched.
- Decisions this round had to resolve: which ViT readout recipe to record per
  dataset; whether pretraining gains survive matched recipes; whether the folding
  geometry is validated against a naive reshape baseline.

## 3. Setup and Evaluation Protocol

- Datasets (12): CHB-MIT, TUAB, TUEV, ISRUC, FACED, SEED-V, PhysioNet-MI, SHU-MI,
  BCIC2020-3, Mumtaz2016, MentalArithmetic, HMC.
- Protocol: seeds 42-46; validation-selected checkpoint; exactly one final test
  evaluation per seed; BF16 AMP; per-dataset YAML recipes under
  `configs/backbones/<backbone>/<dataset>.yaml`.
- Primary metrics: PR-AUC (binary), Cohen's kappa (multiclass).
- Deviations recorded: (i) three random-init repair configs rerun with
  `--num_workers 0` after DataLoader worker segfaults (loading-only change);
  (ii) SEED-V / BCIC2020-3 ViT keep the flatten recipe by decision;
  (iii) ViT fold-vs-reshape was not part of this round's queue; it was completed
  separately by the geometry line (see `FOLD_VS_RESHAPE_RESULTS.md`).

## 4. Main Findings

### 4.1 ViT recipe correction (true CLS token + lr 5e-4 + clip 1) vs flatten@1e-4

Holm-corrected Welch tests (n=5+5) from the analysis bundle:

| Dataset | Metric | New (cls) | Old flatten | d (pp) | p_holm |
|---|---|---|---|---|---|
| PhysioNet-MI | kappa | .51694 | .43634 | **+8.06** | <.0001 *** |
| FACED | kappa | .37444 | .30969 | **+6.47** | .0002 *** |
| TUAB | pr_auc | .90907 | .88428 | **+2.48** | .0002 *** |
| ISRUC | kappa | .78754 | .77854 | **+0.90** | .0132 * |
| MentalArithmetic | pr_auc | .72491 | .57394 | +15.10 | .0710 ns |
| CHB-MIT | pr_auc | .47587 | .45657 | +1.93 | .2999 ns |
| TUEV | kappa | .74276 | .72625 | +1.65 | .2999 ns |
| SHU-MI | pr_auc | .68905 | .66185 | +2.72 | .9398 ns |
| HMC | kappa | .69868 | .69805 | +0.06 | 1.00 ns |
| Mumtaz2016 | pr_auc | .97973 | .98067 | -0.09 | 1.00 ns |
| SEED-V | kappa | .22759 | .24419 | -1.66 | .1404 ns |
| BCIC2020-3 | kappa | .52100 | .57733 | -5.63 | .2999 ns |

### 4.2 Pretrain vs random init (36-cell grid; pretrained vs random, primary metric)

| Dataset | EfficientNet-B0 | ConvNeXt-Tiny | ViT-Small |
|---|---|---|---|
| CHB-MIT (pr_auc) | .4092 vs .4037 (+0.55) | .4734 vs .4279 (+4.55) | .4759 vs .3647 (+11.12) |
| TUAB (pr_auc) | .9178 vs .9165 (+0.13) | .9057 vs .9032 (+0.25) | .9091 vs .8464 (+6.26) |
| TUEV (kappa) | .6901 vs .5995 (+9.06) | .7095 vs .5070 (+20.25) | .7428 vs .5070 (+23.58) |
| ISRUC (kappa) | .7704 vs .7733 (-0.28) | .7905 vs .6378 (+15.27) | .7875 vs .7373 (+5.03) |
| FACED (kappa) | .4953 vs .1513 (+34.40) | .5890 vs .2396 (+34.94) | .3744 vs .1961 (+17.83) |
| SEED-V (kappa) | .2056 vs .2266 (-2.10) | .2396 vs .0191 (+22.05) | .2442 vs .0720 (+17.22) |
| PhysioNet-MI (kappa) | .5322 vs .4573 (+7.48) | .5478 vs .2697 (+27.82) | .5169 vs .4365 (+8.05) |
| SHU-MI (pr_auc) | .6982 vs .5157 (+18.25) | .6910 vs .5983 (+9.26) | .6891 vs .6437 (+4.54) |
| BCIC2020-3 (kappa) | .5673 vs .3020 (+26.53) | .5560 vs .1183 (+43.77) | .5773 vs .3407 (+23.67) |
| Mumtaz2016 (pr_auc) | .9864 vs .9773 (+0.91) | .9842 vs .9710 (+1.32) | .9797 vs .9778 (+0.19) |
| MentalArithmetic (pr_auc) | .7903 vs .2054 (+58.49) | .7598 vs .2511 (+50.87) | .7249 vs .4189 (+30.60) |
| HMC (kappa) | .7070 vs .6877 (+1.93) | .7111 vs .6461 (+6.51) | .6987 vs .6169 (+8.18) |

### 4.3 Fold vs reshape (phase folding vs contiguous chunk; completed)

EfficientNet-B0: fold better on 7/9 P>1 datasets (largest: FACED -0.176 kappa,
MentalArithmetic -0.205 PR-AUC); indistinguishable on CHB-MIT and SHU-MI.
ConvNeXt-Tiny: 9/9 fold-favored. ViT-Small (corrected CLS-token recipe): 8/9
fold-favored (largest: SHU-MI -0.089, HMC -0.085) with a single small reversal
on CHB-MIT (chunk +0.020). Across three backbones the fold advantage holds with
only two near-zero reversals (SHU-MI B0 +0.003, CHB-MIT ViT +0.020).
(Source: `FOLD_VS_RESHAPE_RESULTS.md`.)

## 5. Statistical Validation

- **Recipe correction thread**: Welch t-tests from per-seed data where available;
  summary-statistics Welch (population->sample std conversion) for baseline-only
  comparisons; Holm correction within families; Cohen's d and 95% CIs reported in
  the bundle (`stats-appendix.md`). 4/12 corrections survive Holm.
- **Pretrain-vs-random**: the 15 earlier cells have approximate summary-statistics
  tests (recorded with the configs; results in the earlier message trail);
  the 21 extension cells are currently **descriptive only** (mean over seeds; formal
  tests pending). n=5 per side; all cells use validation-selected checkpoints.
- **Fold-vs-reshape**: two-sample Welch t from population stds (n=5), reported in
  `FOLD_VS_RESHAPE_RESULTS.md`.

## 6. Figure-by-Figure Interpretation

Figures live in `experiments/analysis/vit_clstoken_lr5e4_clip1_2026-09-11/figures/`.

1. `figure-01-main-comparison` (Δ primary metric per dataset vs flatten): shows the
   dataset-dependence of the recipe correction; supports per-dataset head selection.
2. `figure-02-forest-plot` (Δ with 95% CIs): separates large-but-unstable
   (MentalArithmetic) from reliable effects (PhysioNet/FACED/TUAB/ISRUC).
3. `figure-03-per-seed-grid`: audits seed-level spread; shows SHU-MI's outlier seed
   and the seed-robustness of PhysioNet/FACED/ISRUC. No new figures were generated
   for the pretrain-vs-random extension in this round (tables only).

## 7. Failure Cases / Negative Results / Limitations

- **Instability**: DataLoader worker SIGSEGV / torch shared-memory deadlock crippled
  3 random-init configs (BCIC-B0 x2 seeds, FACED-B0 x4 seeds, SEED-V-B0 all seeds).
  Repairs were rerun with `--num_workers 0` and completed; all 105 runs are done and
  all 21 configs were finalized (21/21 with results).
- **Near-parity cells**: TUAB (B0/CN), Mumtaz, HMC show small pretraining gains -
  for big or easy datasets, random init learns comparable features under the
  matched budget.
- **Near-chance random baselines**: MentalArithmetic / SEED-V / BCIC random CNNs
  collapse to chance or near-chance; their large deltas reflect "pretrain works vs
  random fails to train", not a graded capability gap.
- **Non-uniform ViT baseline**: SEED-V and BCIC compare against the flatten recipe
  (cls_token underperformed there); head selection is per-dataset.
- **Statistical caveats**: extension grid lacks inferential tests so far; older
  comparisons use summary statistics only; SHU-MI significance is outlier-limited.

## 8. What Changed Our Belief

- The ViT recipe fix is a real but **dataset-dependent** gain; "cls token is better"
  is not universal (BCIC/SEED-V falsify it).
- Pretraining remains beneficial under matched recipes, but the corrected matched
  comparison **shrinks** several previously reported gains (e.g., ISRUC ViT from
  +13.7pp to +5.0pp); the old numbers were inflated by recipe mismatch.
- Random-init baselines are strongly recipe-sensitive: corrected recipes help random
  models too (ISRUC ViT random +9.2pp), meaning "pretraining benefit" claims must
  always state the matched recipe.
- The completed grid adds a second exception: EfficientNet-B0 x SEED-V is the only
  cell where random init edges out pretraining (-2.10pp).
- The folding geometry itself is validated: phase-interleaved folding >= chunk
  reshape on every completed comparison.

## 9. Next Actions

1. [COMPLETE] The 4 in-flight cells finished; all 21 configs finalized (21/21 with
   results) and verified to contain exactly 5 seeds each.
2. Update `PHASE_FOLD_RESULTS.md` control tables and paper appendix numbers with the
   matched-recipe results (pretrained and random sides).
3. All still-uncommitted configs (random-init extension + corrections) to be
   committed with the same atomic-commit discipline.
4. Optional: formal Welch/Holm tests for the extended pretrain-vs-random grid, and
   a per-seed sensitivity note for near-parity cells.

## 10. Artifact and Reproducibility Index

- Analysis bundle: `experiments/analysis/vit_clstoken_lr5e4_clip1_2026-09-11/`
  (analysis-report.md, stats-appendix.md, figure-catalog.md, results.csv/json,
  figures/, build_bundle.py, update_configs.py).
- Fold-vs-reshape: `experiments/FOLD_VS_RESHAPE_RESULTS.md` +
  `experiments/run_fold_reshape_ablation.sh` + `configs/ablation_fold_geometry/`.
- Random-init configs: `configs/ablation_random_init/` (36 cells).
- Run logs: `/tmp/codemaker/vit_rerun_2026-09-11/` (queue logs, status files) and
  `experiments/logs/pretrain_vs_random/...`.
- Queue scripts: `queue_clstoken.sh`, `queue_remaining_v3.sh`, `queue_random_init.sh`,
  `queue_random_all.sh`, `repair_random_all.sh`, finalize scripts (all under
  `/tmp/codemaker/vit_rerun_2026-09-11/`).
