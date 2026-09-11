# Analysis Report — ViT-Small DINOv3: true cls-token, lr 5e-4, clip 1

Bundle: `experiments/analysis/vit_clstoken_lr5e4_clip1_2026-09-11`
Date: 2026-09-11 · Runs: 12 datasets x 5 seeds (42–46), 60 final test evaluations

## 1. Analysis question

Does the corrected compact head (true CLS token via `feature_aggregation=cls_token`,
i.e. `features[:, 0]` after the final norm) combined with lr=5e-4 and clip=1 improve
the ViT-Small DINOv3 backbone across the 12 downstream datasets, relative to the
previously recorded recipes?

Two premises drove the rerun:
1. The previously recorded "cls_token (gap)" head was verified to be patch-token
   mean pooling (timm `Eva`, `global_pool='avg'`), not the CLS token. The old
   comparisons were therefore mislabeled.
2. The previously recorded ViT learning rate (1e-4, clipping disabled) was suspected
   to be too low, following the TUAB finding that a compact head at lr=5e-4 with
   clip=1 fixed the flatten-head overfitting.

## 2. Protocol and comparison units

- **New runs**: `vit_small_patch16_dinov3.lvd1689m`, each dataset's YAML recipe,
  overridden with `--vision_feature_aggregation cls_token --lr 5e-4 --clip_value 1`.
  Validation-selected checkpoint, one final test evaluation per seed, seeds 42–46.
- **Primary baseline**: previously recorded `flatten@1e-4` YAML results (n=5 each).
- **Secondary (exploratory)**: previously recorded patch-mean@1e-4 head comparisons
  where present (9 datasets; these are the runs that had been labeled "cls_token").
- Unit of analysis: dataset x seed. Primary metric: `pr_auc` (binary tasks) or
  `kappa` (multiclass tasks).

## 3. Headline findings

Against the old **flatten** baseline (Holm-corrected Welch tests, n=5+5, m=12):

| Dataset | Metric | New (mean ± s) | Old flatten | Δ (pp) | p_holm | d | Sig |
|---|---|---|---|---|---|---|---|
| MentalArithmetic | pr_auc | .72491 ± .06991 | .57394 | **+15.10** | .0710 | +2.17 | ns |
| PhysioNet-MI | kappa | .51694 ± .00630 | .43634 | **+8.06** | <.0001 | +11.36 | *** |
| FACED | kappa | .37444 ± .01218 | .30969 | **+6.47** | .0002 | +6.28 | *** |
| SHU-MI | pr_auc | .68905 ± .05115 | .66185 | +2.72 | .9398 | +0.71 | ns |
| TUAB | pr_auc | .90907 ± .00376 | .88428 | +2.48 | .0002 | +6.03 | *** |
| CHB-MIT | pr_auc | .47587 ± .01320 | .45657 | +1.93 | .2999 | +1.39 | ns |
| TUEV | kappa | .74276 ± .01415 | .72625 | +1.65 | .2999 | +1.56 | ns |
| ISRUC | kappa | .78754 ± .00290 | .77854 | +0.90 | .0132 | +3.00 | * |
| HMC | kappa | .69868 ± .00364 | .69805 | +0.06 | 1.0000 | +0.22 | ns |
| Mumtaz2016 | pr_auc | .97973 ± .00412 | .98067 | −0.09 | 1.0000 | −0.30 | ns |
| SEED-V | kappa | .22759 ± .01050 | .24419 | −1.66 | .1404 | −2.02 | ns |
| BCIC2020-3 | kappa | .52100 ± .04602 | .57733 | −5.63 | .2999 | −1.67 | ns |

- **Four datasets improve, three at Holm-corrected significance**: PhysioNet-MI
  (+8.06pp κ), FACED (+6.47pp κ), TUAB (+2.48pp pr_auc), and ISRUC (+0.90pp κ).
- **MentalArithmetic** has the largest raw gain (+15.10pp pr_auc, d=+2.17) but fails
  Holm correction due to high seed variance (σ=.070).
- **Two datasets move backwards**, neither significant after correction:
  BCIC2020-3 (−5.63pp κ, raw p=.050) and SEED-V (−1.66pp κ, raw p=.020).
- **Mumtaz2016 and HMC are at parity.**

Against the previously selected **patch-mean** head (exploratory, m=9): FACED
(+17.14pp, p_holm<.0001) and PhysioNet-MI (+4.58pp, p_holm=.0078) improve further;
TUAB is at parity (−0.23pp); SEED-V is worse than both recorded heads (−1.85pp,
p_holm=.090); the rest are not distinguishable.

## 4. Strongest supported comparisons

1. **PhysioNet-MI** κ .51694 ± .00630 vs .43634 — the largest effect size in the
   family (d=+11.4); all five seeds sit above the old mean.
2. **FACED** κ .37444 ± .01218 vs .30969 — d=+6.28; also +17.1pp over the old
   patch-mean comparison, rewriting the earlier "cls is worse than flatten" conclusion.
3. **TUAB** pr_auc .90907 ± .00376 vs .88428 (d=+6.03) — note the previously reported
   TUAB ViT recipe was the patch-mean head (.91141); the new head is at parity with
   that (−0.23pp), i.e. no regression vs the reported recipe, while remaining far
   above the historical flatten head.
4. **ISRUC** κ .78754 ± .00290 vs .77854 — d=+3.00, tight seeds.

## 5. What did not improve (and why it matters)

- **BCIC2020-3**: flatten remains the best recorded head (κ .57733 vs new .52100).
  The true cls-token route does not rescue this dataset.
- **SEED-V**: both old heads beat the new recipe; this dataset favors flatten /
  patch-mean aggregation.
- **HMC, Mumtaz2016**: parity — the old flatten results already matched the new
  recipe within ±0.1pp.

## 6. Main caveats and blockers

- No per-seed baseline values exist in the records; all tests are **unpaired Welch
  tests from summary statistics** (old population stds converted to sample stds).
- n=5 per group; normality assumptions are not verifiable at this sample size.
- SHU-MI contains one outlier seed (s45: pr_auc .598); excluding it raises the mean
  to .71183 (see appendix). Its non-significance is driven by that seed's variance.
- MentalArithmetic's gain is large but unstable; promising, not confirmed.
- The patch-mean family is exploratory (partially overlapping datasets).

## 7. What changed in the experimental understanding

- The head/lr correction is a **dataset-dependent** win: 4 improvements (3 significant),
  2 regressions (1 near-significant raw), 2 parities — not a uniform upgrade.
- The previously recorded "cls_token" comparisons were mislabeled patch-mean results;
  these are the first true CLS-token evaluations.
- The two-stage readout policy (compact first, flatten fallback) remains the right
  frame; with corrected semantics the compact choice is now: cls_token for
  PhysioNet-MI / FACED / TUAB / ISRUC / MentalArithmetic; flatten for BCIC2020-3;
  flatten or patch-mean for SEED-V.

## 8. Follow-ups (outside this bundle)

- Update the ViT YAML configs and `experiments/PHASE_FOLD_RESULTS.md` with the new
  recorded results (bookkeeping step).
- Paper tables that carry the old "cls_token" labels need the corrected semantics.
