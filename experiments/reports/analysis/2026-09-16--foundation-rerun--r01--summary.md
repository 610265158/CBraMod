---
type: results-report
date: 2026-09-16
experiment_line: foundation-rerun
round: 1
purpose: summary
status: active
source_artifacts:
  - experiments/reports/analysis/analysis-report.md
  - experiments/reports/analysis/stats-appendix.md
  - experiments/reports/analysis/figure-catalog.md
  - experiments/reports/foundation_rerun_appendix.md
  - experiments/reports/foundation_rerun_results.csv
obsidian_write_back: not attempted (repository is not bound to an Obsidian project KB)
---

# Foundation Rerun / Round 1 / Summary / 2026-09-16

## 1. Executive Summary

- Completed a matched-pipeline rerun of CBraMod and REVE on FACED, TUAB and HMC:
  **30/30 runs** (2 models x 3 datasets x 5 seeds), zero failures, ~13.9 h on one RTX 4090.
- Protocol: this repository's per-dataset recipe **unchanged**; the only adapted
  hyperparameter is the learning rate (1e-4); inputs are normalized by the dataset
  loaders with each model's released convention (no clipping); validation-selected
  checkpoint and one final test evaluation per seed (seeds 42-46).
- Headline numbers (test-time, mean +/- population std over 5 seeds):
  - **CBraMod / FACED reproduces the published reference**: kappa .4961 +/- .0124 vs
    published .5041 +/- .0122 (delta -0.0080); the 95% CI [.4790, .5133] overlaps the
    published interval.
  - **REVE is systematically lower**: FACED kappa .4190 vs .5080 (delta -0.0890);
    TUAB PR-AUC .8901 vs .9281 (delta -0.0380); HMC kappa .6831 vs .6982 (delta -0.0151).
  - **CBraMod on TUAB is also lower**: PR-AUC .8632 vs .9221 (delta -0.0589).
  - CBraMod has no published HMC reference; its rerun is a supplementary cell
    (kappa .6758 +/- .0026).
- Decision: keep the published numbers in the main comparison but label the
  foundation-model rows as **contextual references** and report the rerun deviations
  in the appendix (reviewer-provided framing). The suggested main-text sentence is
  included in `experiments/reports/foundation_rerun_appendix.md`.

## 2. Experiment Identity and Decision Context

- Trigger: reviewer asked to assess comparability of published foundation-model
  references via a small, strictly aligned rerun on representative datasets.
- Decision at stake: (a) can the unified table stay apples-to-apples, and (b) do the
  paper's "vision model wins" statements survive a controlled rerun of the baselines?
- Answer after this round: CBraMod on FACED is reproducible within noise; REVE
  reruns land 1.5-8.9 points below published under this recipe; the vision-vs-rerun
  comparison strengthens FACED and flips TUAB toward the vision models, while HMC
  remains a close case.

## 3. Setup and Evaluation Protocol

- Chain: `experiments/run_downstream.sh` -> `experiments/downstream_11.py` ->
  `finetune_main.py` -> repository loaders + `finetune_trainer.Trainer` +
  `finetune_evaluator.Evaluator`. No dataset file is re-preprocessed; splits are the
  inherited CBraMod/REVE benchmark partitions.
- Input normalization (loader level): CBraMod microvolt / 100; REVE microvolt / scale_factor
  (FACED 1000, TUAB 100, HMC 10); no clipping for both. Vision runs keep clip +/-1024
  and divide-by-32 (probe verified).
- Selection: validation kappa (multiclass) / PR-AUC (binary); one final test per seed.
- Documented deviations from the released foundation-model procedures: no linear-probe
  warm start, no mixup, no StableAdamW, no LoRA and no model souping for REVE; uniform
  learning rate for CBraMod (released multi-LR not used); REVE TUAB fed the 16-channel
  bipolar arrays instead of its released 21-channel memmap.
- Runtime: queue started 2026-09-15 20:13, finished 2026-09-16 10:05. Per-cell pace:
  REVE FACED ~7-14 min/seed; REVE/CBraMod TUAB ~30-48 min/seed; HMC ~30 min/seed.

## 4. Main Findings

| Dataset | Model | n | Rerun primary | Rerun BA | Published | Delta (primary) |
| --- | --- | ---: | --- | --- | --- | ---: |
| FACED | CBraMod | 5 | kappa .4961 +/- .0124 | .5551 +/- .0107 | .5041 +/- .0122 | **-0.0080** |
| FACED | REVE | 5 | kappa .4190 +/- .0063 | .4850 +/- .0062 | .5080 +/- .0191 | **-0.0890** |
| TUAB | CBraMod | 5 | PR-AUC .8632 +/- .0056 | .7821 +/- .0099 | .9221 | -0.0589 |
| TUAB | REVE | 5 | PR-AUC .8901 +/- .0114 | .8052 +/- .0072 | .9281 +/- .0009 | -0.0380 |
| HMC | REVE | 5 | kappa .6831 +/- .0046 | .7337 +/- .0031 | .6982 +/- .0078 | -0.0151 |
| HMC | CBraMod | 5 | kappa .6758 +/- .0026 | .7279 +/- .0038 | not reported | -- |

Context against the paper's vision rows: FACED (B0 kappa .4953, ConvNeXt .5891)
-> the rerun CBraMod ties B0 and the rerun REVE falls well below; TUAB (B0 PR-AUC
.9178, ConvNeXt .9057) -> both rerun baselines fall below the vision models;
HMC (B0 kappa .7070, ConvNeXt .7111) -> rerun REVE .6831 and CBraMod .6758 remain
below, consistent with the published ordering.

## 5. Statistical Validation

- n=5 per group; population standard deviation; t-based 95% CI (df=4).
- Interval comparisons against the published mean +/- 2 std:
  - FACED / CBraMod: rerun CI [.4790, .5133] vs published [.4797, .5285] -> overlap.
  - FACED / REVE: rerun CI [.4103, .4277] vs published [.4698, .5462] -> no overlap.
  - HMC / REVE: rerun CI [.6767, .6895] vs published [.6826, .7138] -> overlap.
  - TUAB / REVE: rerun CI [.8742, .9059] vs published [.9263, .9299] -> no overlap.
  - TUAB / CBraMod: published std not available; interval comparison blocked.
- Blocker statement: the published std is run-to-run variability from the authors'
  (unknown) aggregation, not the standard error of the comparison; no seed-level
  pairing exists, so no significance test is claimed.

## 6. Figure-by-Figure Interpretation

- `figure-01-rerun-vs-published` (figures/): grouped bars of the primary test metric
  per (dataset, model) with population-std error bars and delta annotations.
  Observation: the CBraMod/FACED bar pair nearly coincides; all REVE bars sit visibly
  lower; annotations quantify the deltas. Decision implication: report rerun vs
  published deviations explicitly instead of asserting exact reproducibility.
- `figure-02-val-curves` (figures/): per-epoch validation curves (thin = seed, thick =
  mean) per dataset and model. Observation: runs early-stop on a validation plateau
  (e.g. FACED REVE best epoch ~7); seed spread is tight for both models. Decision
  implication: the deviations are systematic for REVE, not instability; if closer
  reproduction is required, the training procedure (warm start / schedule), not seed
  count, is the lever.

## 7. Failure Cases / Negative Results / Limitations

- REVE / FACED is the largest deviation (-8.9 points); the rerun does not reproduce
  the published value under this recipe.
- TUAB is below published for both models; under the rerun framing the vision models
  lead on PR-AUC.
- HMC / CBraMod has no published reference; it is supplementary and cannot be
  compared against the original paper.
- The deviations are documented, not individually ablated (which of lp warm start,
  mixup, StableAdamW, LoRA or souping accounts for the REVE gap remains unknown).
- Single-GPU campaign; per-cell seeds fixed at 42-46 for comparability with the
  repository's protocol.

## 8. What Changed Our Belief

- CBraMod's published references are reproducible inside this harness at least on
  FACED; the "unfair transcription" concern is substantially weakened there.
- REVE's published numbers depend on procedure/aggregation details beyond the released
  default script reachable in this pipeline; the rerun should be presented as a
  contextual reference with explicit deviations.
- Under the rerun protocol the vision models' advantage grows on FACED and appears on
  TUAB; the published "REVE strongest on TUAB" statement only holds for the published
  (contextual) row.

## 9. Next Actions

- Fold `foundation_rerun_appendix.md` (summary table, per-run detail, deviations,
  suggested main-text sentence) and the analysis bundle into the paper/rebuttal.
- If reviewers demand closer REVE reproduction: schedule a fidelity arm
  (lp -> ft warm start + mixup + StableAdamW + optional souping) as a separate,
  documented experiment; optionally obtain the 21-channel TUAB memmap for REVE.
- Freeze this round's artifacts; keep the queue/collector/analyzer scripts for reruns.

## 10. Artifact and Reproducibility Index

- Tables: `experiments/reports/foundation_rerun_appendix.md`,
  `experiments/reports/foundation_rerun_results.csv`
- Analysis bundle: `experiments/reports/analysis/` (analysis-report.md,
  stats-appendix.md, figure-catalog.md, figures/figure-01*, figures/figure-02*)
- Runners: `experiments/run_foundation_rerun.sh` (queue, .done-resumable),
  `experiments/watch_foundation_rerun.sh` (watchdog), `experiments/collect_foundation_results.py`,
  `experiments/analyze_foundation_rerun.py`
- Model code: `models/foundation_cbramod.py`, `models/foundation_reve.py`, vendored
  backbones (`models/cbramod.py`, `models/criss_cross_transformer.py`,
  `models/reve_encoder.py`, `models/reve_backbone.py`), specs `configs/foundation.py`
- Logs/checkpoints: `experiments/logs/foundation_rerun/`,
  `experiments/checkpoints/foundation_rerun/` (best.pth per run + `.done` sentinels)
- Planning: `task_plan.md`, `notes.md` (repository root)
