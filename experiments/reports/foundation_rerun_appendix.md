# Foundation-model matched-pipeline rerun (CBraMod / REVE)

Protocol: per-dataset recipe from `configs/downstream.py` (unchanged); the only adapted hyperparameter is the learning rate (`--lr 1e-4`). Inputs are normalized by the dataset loaders with each model's released convention (CBraMod microvolt / 100; REVE per-dataset `scale_factor`; no clipping). Checkpoints are selected on the validation metric and evaluated once on test per seed.

Models: CBraMod (4.92M parameters, full fine-tuning of the released checkpoint) and REVE-Base (69.19M parameters, full fine-tuning). No encoder is frozen.

### Deviations and provenance

- Training recipe: this repository's per-dataset configuration is used unchanged; the only adapted hyperparameter is the learning rate (1e-4 for both models).
- CBraMod: the released recipe uses a multi-LR setting (encoder 1e-4, head 1e-3*sqrt(batch/256)); this rerun uses one uniform learning rate. AdamW in both cases.
- REVE: the released procedure warm-starts with a linear probe before full fine-tuning, applies mixup and StableAdamW, and the paper additionally describes LoRA and model souping; this rerun is a single fine-tuning stage per seed with AdamW and no mixup/LoRA/souping.
- Inputs: identical pre-processed benchmark arrays (inherited filters and splits); each loader applies the model's released normalization (CBraMod microvolt/100; REVE per-dataset scale_factor; no clipping). REVE's released TUAB pipeline uses a 21-channel memmap; this rerun feeds it the same 16-channel bipolar TUAB arrays as CBraMod.
- Protocol: validation-selected checkpoint, one final test per seed, seeds 42-46, population std.

## Suggested main-text sentence

> To assess the comparability of published foundation-model references, we additionally reran CBraMod and REVE on three representative datasets under the inherited partitions and final-test protocol. The rerun results and deviations from published values are reported in Appendix X.

## Summary (completed seeds, mean +/- population standard deviation)

| Dataset | Model | Seeds | Rerun BA | Rerun primary | Published | Difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FACED | cbramod | 5 | 0.5551 +/- 0.0107 | 0.4961 +/- 0.0124 (kappa) | 0.5041 +/- 0.0122 | -0.0080 |
| FACED | reve | 5 | 0.4850 +/- 0.0062 | 0.4190 +/- 0.0063 (kappa) | 0.5080 +/- 0.0191 | -0.0890 |
| HMC | cbramod | 5 | 0.7279 +/- 0.0038 | 0.6758 +/- 0.0026 (kappa) | -- | -- |
| HMC | reve | 5 | 0.7337 +/- 0.0031 | 0.6831 +/- 0.0046 (kappa) | 0.6982 +/- 0.0078 | -0.0151 |
| TUAB | cbramod | 5 | 0.7821 +/- 0.0099 | 0.8632 +/- 0.0056 (pr_auc) | 0.9221 | -0.0589 |
| TUAB | reve | 5 | 0.8052 +/- 0.0072 | 0.8901 +/- 0.0114 (pr_auc) | 0.9281 +/- 0.0009 | -0.0380 |

## Per-run detail

| Dataset | Model | Seed | Status | LR | Batch | Epochs | Best epoch | Val sel | Val best | Test BA | Test primary | Test secondary | Published | Difference | Checkpoint |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FACED | cbramod | seed42 | done | 0.0001 | 32 | 50 | 47 | kappa | 0.5446 | 0.5606 | 0.5032 | 0.5615 | 0.5041 | -0.0009 | experiments/checkpoints/foundation_rerun/cbramod/seed42/faced/best.pth |
| FACED | cbramod | seed43 | done | 0.0001 | 32 | 49 | 39 | kappa | 0.5430 | 0.5423 | 0.4820 | 0.5441 | 0.5041 | -0.0221 | experiments/checkpoints/foundation_rerun/cbramod/seed43/faced/best.pth |
| FACED | cbramod | seed44 | done | 0.0001 | 32 | 50 | 43 | kappa | 0.5488 | 0.5632 | 0.5047 | 0.5619 | 0.5041 | 0.0006 | experiments/checkpoints/foundation_rerun/cbramod/seed44/faced/best.pth |
| FACED | cbramod | seed45 | done | 0.0001 | 32 | 48 | 38 | kappa | 0.5387 | 0.5423 | 0.4805 | 0.5410 | 0.5041 | -0.0236 | experiments/checkpoints/foundation_rerun/cbramod/seed45/faced/best.pth |
| FACED | cbramod | seed46 | done | 0.0001 | 32 | 49 | 39 | kappa | 0.5471 | 0.5672 | 0.5102 | 0.5683 | 0.5041 | 0.0061 | experiments/checkpoints/foundation_rerun/cbramod/seed46/faced/best.pth |
| HMC | cbramod | seed42 | done | 0.0001 | 32 | 15 | 5 | kappa | 0.6975 | 0.7218 | 0.6714 | 0.7402 | -- | -- | experiments/checkpoints/foundation_rerun/cbramod/seed42/hmc/best.pth |
| HMC | cbramod | seed43 | done | 0.0001 | 32 | 16 | 6 | kappa | 0.7013 | 0.7287 | 0.6767 | 0.7471 | -- | -- | experiments/checkpoints/foundation_rerun/cbramod/seed43/hmc/best.pth |
| HMC | cbramod | seed44 | done | 0.0001 | 32 | 17 | 7 | kappa | 0.7004 | 0.7337 | 0.6792 | 0.7493 | -- | -- | experiments/checkpoints/foundation_rerun/cbramod/seed44/hmc/best.pth |
| HMC | cbramod | seed45 | done | 0.0001 | 32 | 15 | 5 | kappa | 0.7028 | 0.7279 | 0.6770 | 0.7469 | -- | -- | experiments/checkpoints/foundation_rerun/cbramod/seed45/hmc/best.pth |
| HMC | cbramod | seed46 | done | 0.0001 | 32 | 15 | 5 | kappa | 0.6985 | 0.7273 | 0.6744 | 0.7451 | -- | -- | experiments/checkpoints/foundation_rerun/cbramod/seed46/hmc/best.pth |
| TUAB | cbramod | seed42 | done | 0.0001 | 32 | 5 | 1 | pr_auc | 0.8939 | 0.7698 | 0.8660 | 0.8567 | 0.9221 | -0.0561 | experiments/checkpoints/foundation_rerun/cbramod/seed42/tuab/best.pth |
| TUAB | cbramod | seed43 | done | 0.0001 | 32 | 5 | 1 | pr_auc | 0.9110 | 0.7936 | 0.8674 | 0.8674 | 0.9221 | -0.0547 | experiments/checkpoints/foundation_rerun/cbramod/seed43/tuab/best.pth |
| TUAB | cbramod | seed44 | done | 0.0001 | 32 | 5 | 1 | pr_auc | 0.8919 | 0.7732 | 0.8530 | 0.8467 | 0.9221 | -0.0691 | experiments/checkpoints/foundation_rerun/cbramod/seed44/tuab/best.pth |
| TUAB | cbramod | seed45 | done | 0.0001 | 32 | 5 | 5 | pr_auc | 0.8943 | 0.7935 | 0.8680 | 0.8671 | 0.9221 | -0.0541 | experiments/checkpoints/foundation_rerun/cbramod/seed45/tuab/best.pth |
| TUAB | cbramod | seed46 | done | 0.0001 | 32 | 5 | 1 | pr_auc | 0.9029 | 0.7803 | 0.8614 | 0.8553 | 0.9221 | -0.0607 | experiments/checkpoints/foundation_rerun/cbramod/seed46/tuab/best.pth |
| FACED | reve | seed42 | done | 0.0001 | 32 | 17 | 7 | kappa | 0.4621 | 0.4930 | 0.4259 | 0.4892 | 0.5080 | -0.0821 | experiments/checkpoints/foundation_rerun/reve/seed42/faced/best.pth |
| FACED | reve | seed43 | done | 0.0001 | 32 | 22 | 12 | kappa | 0.4342 | 0.4856 | 0.4203 | 0.4869 | 0.5080 | -0.0877 | experiments/checkpoints/foundation_rerun/reve/seed43/faced/best.pth |
| FACED | reve | seed44 | done | 0.0001 | 32 | 46 | 36 | kappa | 0.4292 | 0.4760 | 0.4104 | 0.4789 | 0.5080 | -0.0976 | experiments/checkpoints/foundation_rerun/reve/seed44/faced/best.pth |
| FACED | reve | seed45 | done | 0.0001 | 32 | 22 | 12 | kappa | 0.4511 | 0.4898 | 0.4252 | 0.4958 | 0.5080 | -0.0828 | experiments/checkpoints/foundation_rerun/reve/seed45/faced/best.pth |
| FACED | reve | seed46 | done | 0.0001 | 32 | 34 | 24 | kappa | 0.4355 | 0.4804 | 0.4132 | 0.4766 | 0.5080 | -0.0948 | experiments/checkpoints/foundation_rerun/reve/seed46/faced/best.pth |
| HMC | reve | seed42 | done | 0.0001 | 32 | 14 | 4 | kappa | 0.7118 | 0.7349 | 0.6834 | 0.7515 | 0.6982 | -0.0148 | experiments/checkpoints/foundation_rerun/reve/seed42/hmc/best.pth |
| HMC | reve | seed43 | done | 0.0001 | 32 | 14 | 4 | kappa | 0.7100 | 0.7312 | 0.6794 | 0.7476 | 0.6982 | -0.0189 | experiments/checkpoints/foundation_rerun/reve/seed43/hmc/best.pth |
| HMC | reve | seed44 | done | 0.0001 | 32 | 15 | 5 | kappa | 0.7084 | 0.7310 | 0.6775 | 0.7474 | 0.6982 | -0.0207 | experiments/checkpoints/foundation_rerun/reve/seed44/hmc/best.pth |
| HMC | reve | seed45 | done | 0.0001 | 32 | 13 | 3 | kappa | 0.7086 | 0.7393 | 0.6908 | 0.7563 | 0.6982 | -0.0074 | experiments/checkpoints/foundation_rerun/reve/seed45/hmc/best.pth |
| HMC | reve | seed46 | done | 0.0001 | 32 | 16 | 6 | kappa | 0.7075 | 0.7322 | 0.6842 | 0.7509 | 0.6982 | -0.0140 | experiments/checkpoints/foundation_rerun/reve/seed46/hmc/best.pth |
| TUAB | reve | seed42 | done | 0.0001 | 32 | 5 | 1 | pr_auc | 0.9167 | 0.8131 | 0.8949 | 0.8968 | 0.9281 | -0.0332 | experiments/checkpoints/foundation_rerun/reve/seed42/tuab/best.pth |
| TUAB | reve | seed43 | done | 0.0001 | 32 | 5 | 2 | pr_auc | 0.9212 | 0.8102 | 0.9042 | 0.8993 | 0.9281 | -0.0239 | experiments/checkpoints/foundation_rerun/reve/seed43/tuab/best.pth |
| TUAB | reve | seed44 | done | 0.0001 | 32 | 5 | 1 | pr_auc | 0.9215 | 0.8055 | 0.8970 | 0.8900 | 0.9281 | -0.0311 | experiments/checkpoints/foundation_rerun/reve/seed44/tuab/best.pth |
| TUAB | reve | seed45 | done | 0.0001 | 32 | 5 | 5 | pr_auc | 0.8980 | 0.7922 | 0.8724 | 0.8657 | 0.9281 | -0.0557 | experiments/checkpoints/foundation_rerun/reve/seed45/tuab/best.pth |
| TUAB | reve | seed46 | done | 0.0001 | 32 | 5 | 3 | pr_auc | 0.9127 | 0.8050 | 0.8818 | 0.8809 | 0.9281 | -0.0463 | experiments/checkpoints/foundation_rerun/reve/seed46/tuab/best.pth |
