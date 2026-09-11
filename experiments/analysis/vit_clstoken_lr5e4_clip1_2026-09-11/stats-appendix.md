# Stats Appendix — ViT-Small cls-token rerun (lr 5e-4, clip 1)

## 1. Data sources

- **New runs** (60): per-seed final test evaluations parsed from the run queue logs
  (`clstoken_*.queue.log`, `rem_*.queue.log`), seeds 42–46, one test per seed.
- **Old flatten / patch-mean**: means and population stds (n=5) recorded in the
  dataset YAMLs under `results` / `head_comparison` (source of truth).
- Parsed tables: `results.json`, `results.csv` (primary metric, with tests),
  `results_all_metrics.csv` (all metrics, descriptive only).

## 2. Descriptive statistics (primary metric, per-seed detail)

| Dataset | Metric | New mean ± s (sample) | Per-seed | Old flatten mean ± σ_pop | Old patch-mean |
|---|---|---|---|---|---|
| MentalArithmetic | pr_auc | .72491 ± .06991 | .61847/.81047/.74922/.71085/.73554 | .57394 ± .06172 | .69924 ± .03928 |
| PhysioNet-MI | kappa | .51694 ± .00630 | .51486/.51267/.51190/.52744/.51782 | .43634 ± .00698 | .47110 ± .01308 |
| FACED | kappa | .37444 ± .01218 | .35489/.37677/.37184/.38556/.38313 | .30969 ± .00717 | .20305 ± .00720 |
| SHU-MI | pr_auc | .68905 ± .05115 | .70858/.70825/.71056/.59794/.71991 | .66185 ± .01610 | .66889 ± .00318 |
| TUAB | pr_auc | .90907 ± .00376 | .90571/.90542/.91439/.91096/.90887 | .88428 ± .00397 | .91141 ± .00188 |
| CHB-MIT | pr_auc | .47587 ± .01320 | .46078/.48653/.48465/.46210/.48528 | .45657 ± .01308 | .46393 ± .03429 |
| TUEV | kappa | .74276 ± .01415 | .75002/.74863/.75923/.72958/.72636 | .72625 ± .00426 | — |
| ISRUC | kappa | .78754 ± .00290 | .78945/.78425/.79130/.78532/.78738 | .77854 ± .00277 | .78255 ± .00352 |
| HMC | kappa | .69868 ± .00364 | .70257/.69799/.69534/.69512/.70236 | .69805 ± .00157 | — |
| Mumtaz2016 | pr_auc | .97973 ± .00412 | .97521/.98484/.97639/.97929/.98290 | .98067 ± .00161 | — |
| SEED-V | kappa | .22759 ± .01050 | .21476/.23658/.23910/.21965/.22786 | .24419 ± .00450 | .24613 ± .00558 |
| BCIC2020-3 | kappa | .52100 ± .04602 | .55833/.49333/.54500/.55500/.45333 | .57733 ± .01129 | .50267 ± .02131 |

## 3. Inference: choices, assumptions, results

**Test choice.** Unpaired Welch t-test computed from summary statistics
(`scipy.stats` t distribution; two-sided, α=.05). A paired test would be the correct
choice, but per-seed baseline values are not recorded — only means and population
stds — so pairing is impossible. This is the primary blocker of the bundle.

**Std conversion.** YAML stds are population stds (n=5). Before testing/CI/effect-size
computation they are converted to sample stds: s_sample = s_pop × sqrt(n/(n−1))
(×1.118). New-run stds are sample stds computed from per-seed values (ddof=1).

**Effect size.** Cohen's d = Δ / sqrt((s₁² + s₂²)/2).

**95% CI** on the mean difference: Δ ± t₀.₉₇₅,df_welch × SE, SE = sqrt(s₁²/n + s₂²/n).

**Multiple comparisons.** Holm correction, applied separately within the flatten
family (m=12, confirmatory) and the patch-mean family (m=9, exploratory).

### Flatten family (confirmatory)

| Dataset | Δ (pp) | t | df | p | p_holm | d | 95% CI (pp) | verdict |
|---|---|---|---|---|---|---|---|---|
| PhysioNet-MI | +8.06 | +17.96 | 7.66 | <.0001 | <.0001 | +11.36 | ±1.04 | *** |
| FACED | +6.47 | +9.93 | 6.92 | <.0001 | .0002 | +6.28 | ±1.55 | *** |
| TUAB | +2.48 | +9.53 | 7.79 | <.0001 | .0002 | +6.03 | ±0.60 | *** |
| ISRUC | +0.90 | +4.75 | 7.96 | .0015 | .0132 | +3.00 | ±0.44 | * |
| MentalArithmetic | +15.10 | +3.44 | 8.00 | .0089 | .0710 | +2.17 | ±10.13 | ns |
| SEED-V | −1.66 | −3.19 | 5.74 | .0201 | .1404 | −2.02 | ±1.29 | ns |
| CHB-MIT | +1.93 | +2.19 | 7.92 | .0602 | .2999 | +1.39 | ±2.04 | ns |
| TUEV | +1.65 | +2.47 | 4.89 | .0574 | .2999 | +1.56 | ±1.73 | ns |
| BCIC2020-3 | −5.63 | −2.64 | 4.60 | .0500 | .2999 | −1.67 | ±5.63 | ns |
| SHU-MI | +2.72 | +1.12 | 4.98 | .3133 | .9398 | +0.71 | ±6.24 | ns |
| Mumtaz2016 | −0.09 | −0.47 | 5.47 | .6569 | 1.0000 | −0.30 | ±0.50 | ns |
| HMC | +0.06 | +0.35 | 5.77 | .7413 | 1.0000 | +0.22 | ±0.45 | ns |

### Patch-mean family (exploratory)

| Dataset | Δ vs patch-mean (pp) | p | p_holm |
|---|---|---|---|
| FACED | +17.14 | <.0001 | <.0001 |
| PhysioNet-MI | +4.58 | .0010 | .0078 |
| MentalArithmetic | +2.57 | .5102 | 1.0000 |
| SHU-MI | +2.02 | .4286 | 1.0000 |
| BCIC2020-3 | +1.83 | .4591 | 1.0000 |
| CHB-MIT | +1.19 | .5398 | 1.0000 |
| ISRUC | +0.50 | .0546 | .3277 |
| TUAB | −0.23 | .2681 | 1.0000 |
| SEED-V | −1.85 | .0129 | .0901 |

## 4. Sensitivity notes

- **SHU-MI outlier.** Seed 45 scored pr_auc .598 vs .708–.720 for the other seeds;
  excluding it, the 4-seed mean is .71183 (+5.0pp vs old flatten) and the variance
  drops from σ=.051 to ≈.005. The non-significant Holm result is an artifact of this
  single seed; treat SHU-MI as "directionally positive, unresolved".
- **MentalArithmetic variance.** New σ=.070 vs old σ=.062; the +15.1pp raw gain has a
  CI of ±10.1pp. Direction is positive in 5/5 seeds, but the effect is not confirmed
  at family-wise level.
- **TUAB baseline dependence.** +2.48pp is significant vs the flatten baseline, but
  the reported TUAB ViT recipe was patch-mean; against it the new head is −0.23pp
  (ns). Both statements are true and are reported together.

## 5. Blockers and limitations

1. No per-seed baselines → all comparisons are unpaired summary-statistics tests.
2. Population-to-sample std conversion assumes the YAML stds are exact population
   stds of the 5 recorded seeds (as documented).
3. n=5; normality assumptions untestable; Welch robust to unequal variances only.
4. Old results were produced under the same frozen protocol but in an earlier
   environment; drift cannot be excluded.
5. The patch-mean family reuses overlapping data points; its p-values are exploratory.

## 6. Reference files

- `results.json` — full primary-metric statistics (means, stds, t, df, p, p_holm, d, CI, per-seed)
- `results.csv` — same in CSV
- `results_all_metrics.csv` — new-vs-old descriptive table for all metrics (36 rows)
