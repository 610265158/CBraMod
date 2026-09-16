# Rerun analysis: CBraMod / REVE matched-pipeline rerun

Analysis question: under this repository's own training recipe (learning rate adapted to 1e-4), how do the reruns of CBraMod and REVE compare with the published reference values on the shared benchmark partitions?

## Key findings

- **FACED / cbramod** (n=5): kappa = 0.4961 +/- 0.0124 (population std) vs published 0.5041 +/- 0.0122 -> delta -0.0080 (95% CI [0.4790, 0.5133]).
- **FACED / reve** (n=5): kappa = 0.4190 +/- 0.0063 (population std) vs published 0.5080 +/- 0.0191 -> delta -0.0890 (95% CI [0.4103, 0.4277]).
- **HMC / cbramod** (n=5): kappa = 0.6758 +/- 0.0026 (population std); no published reference for this dataset.
- **HMC / reve** (n=5): kappa = 0.6831 +/- 0.0046 (population std) vs published 0.6982 +/- 0.0078 -> delta -0.0151 (95% CI [0.6767, 0.6895]).
- **TUAB / cbramod** (n=5): pr_auc = 0.8632 +/- 0.0056 (population std) vs published 0.9221 +/- n/a -> delta -0.0589 (95% CI [0.8554, 0.8709]).
- **TUAB / reve** (n=5): pr_auc = 0.8901 +/- 0.0114 (population std) vs published 0.9281 +/- 0.0009 -> delta -0.0380 (95% CI [0.8742, 0.9059]).

## Caveats and blockers

- The published reference is a mean +/- run-to-run std from the original authors (their aggregation details, e.g. model souping, are not fully known); the rerun is one val-selected run per seed under this repository's protocol, so no paired significance test is valid. We report intervals and standardized gaps instead.
- Recipe deviations are documented in experiments/reports/foundation_rerun_appendix.md (no linear-probe warm start, no mixup/StableAdamW/LoRA/souping for REVE; uniform LR for CBraMod).
- Dataset coverage is incomplete while the grid runs; groups with n<2 seeds have no interval.

