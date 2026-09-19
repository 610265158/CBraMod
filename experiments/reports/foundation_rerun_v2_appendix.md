# Foundation-model matched-pipeline rerun (CBraMod / REVE)

Protocol: the v2 unified recipe (learning rate 1e-4, warmup 3 epochs at factor 0.1, EMA 0.995, weight decay 5e-4, gradient clipping 1.0, early stop 10) with each dataset's epochs and selection metric from `configs/downstream.py`. Inputs are normalized by the dataset loaders with each model's released convention (CBraMod microvolt / 100; REVE per-dataset `scale_factor`; no clipping). Checkpoints are selected on the validation metric and evaluated once on test per seed (seeds 42-46).

Models: CBraMod (4.92M parameters, full fine-tuning of the released checkpoint) and REVE-Base (69.19M parameters, full fine-tuning). No encoder is frozen.

### Deviations and provenance

- Training recipe: the v2 unified recipe is applied to both models; the learning rate (1e-4) remains the only per-model adaptation.
- CBraMod: the released recipe uses a multi-LR setting (encoder 1e-4, head 1e-3*sqrt(batch/256)); this rerun uses one uniform learning rate. AdamW in both cases.
- REVE: the released procedure warm-starts with a linear probe before full fine-tuning, applies mixup and StableAdamW, and the paper additionally describes LoRA and model souping; this rerun is a single fine-tuning stage per seed with AdamW and no mixup/LoRA/souping.
- Inputs: identical pre-processed benchmark arrays (inherited filters and splits); each loader applies the model's released normalization (CBraMod microvolt/100; REVE per-dataset scale_factor; no clipping). REVE's released TUAB pipeline uses a 21-channel memmap; this rerun feeds it the same 16-channel bipolar TUAB arrays as CBraMod.
- Protocol: validation-selected checkpoint, one final test per seed, seeds 42-46, population std.

## Suggested main-text sentence

> To assess the comparability of published foundation-model references, we additionally reran CBraMod and REVE on five datasets under the inherited partitions and final-test protocol. The rerun results and deviations from published values are reported in Appendix X.

## Summary (completed seeds, mean +/- population standard deviation)

| Dataset | Model | Seeds | Rerun BA | Rerun primary | Published | Difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FACED | cbramod | 5 | 0.5531 +/- 0.0030 | 0.4939 +/- 0.0035 (kappa) | 0.5041 +/- 0.0122 | -0.0102 |
| FACED | reve | 5 | 0.5127 +/- 0.0249 | 0.4505 +/- 0.0274 (kappa) | 0.5080 +/- 0.0191 | -0.0575 |
| HMC | cbramod | 5 | 0.7287 +/- 0.0023 | 0.6752 +/- 0.0023 (kappa) | -- | -- |
| HMC | reve | 5 | 0.7343 +/- 0.0021 | 0.6813 +/- 0.0035 (kappa) | 0.6982 +/- 0.0078 | -0.0169 |
| Mumtaz2016 | cbramod | 5 | 0.9008 +/- 0.0078 | 0.9767 +/- 0.0109 (pr_auc) | 0.9923 +/- 0.0032 | -0.0156 |
| Mumtaz2016 | reve | 5 | 0.9253 +/- 0.0152 | 0.9912 +/- 0.0056 (pr_auc) | 0.9961 +/- 0.0013 | -0.0049 |
| PhysioNet-MI | cbramod | 5 | 0.6183 +/- 0.0070 | 0.4910 +/- 0.0094 (kappa) | 0.5222 +/- 0.0169 | -0.0312 |
| PhysioNet-MI | reve | 5 | 0.6916 +/- 0.0057 | 0.5888 +/- 0.0076 (kappa) | 0.5306 +/- 0.0187 | 0.0582 |
| TUAB | cbramod | 5 | 0.8066 +/- 0.0026 | 0.8923 +/- 0.0026 (pr_auc) | 0.9221 | -0.0298 |
| TUAB | reve | 5 | 0.8064 +/- 0.0053 | 0.8833 +/- 0.0062 (pr_auc) | 0.9281 +/- 0.0009 | -0.0448 |

## Per-run detail

| Dataset | Model | Seed | Status | LR | Batch | Epochs | Best epoch | Val sel | Val best | Test BA | Test primary | Test secondary | Published | Difference | Checkpoint |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FACED | cbramod | seed42 | done | 0.0001 | 32 | 50 | 43 | kappa | 0.5554 | 0.5515 | 0.4934 | 0.5524 | 0.5041 | -0.0107 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed42/faced/best.pth |
| FACED | cbramod | seed43 | done | 0.0001 | 32 | 50 | 40 | kappa | 0.5431 | 0.5562 | 0.4984 | 0.5583 | 0.5041 | -0.0057 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed43/faced/best.pth |
| FACED | cbramod | seed44 | done | 0.0001 | 32 | 48 | 38 | kappa | 0.5568 | 0.5535 | 0.4935 | 0.5539 | 0.5041 | -0.0106 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed44/faced/best.pth |
| FACED | cbramod | seed45 | done | 0.0001 | 32 | 34 | 24 | kappa | 0.5606 | 0.5560 | 0.4964 | 0.5558 | 0.5041 | -0.0077 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed45/faced/best.pth |
| FACED | cbramod | seed46 | done | 0.0001 | 32 | 43 | 33 | kappa | 0.5526 | 0.5483 | 0.4880 | 0.5485 | 0.5041 | -0.0161 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed46/faced/best.pth |
| HMC | cbramod | seed42 | done | 0.0001 | 32 | 19 | 9 | kappa | 0.7007 | 0.7294 | 0.6739 | 0.7469 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed42/hmc/best.pth |
| HMC | cbramod | seed43 | done | 0.0001 | 32 | 17 | 7 | kappa | 0.7029 | 0.7251 | 0.6730 | 0.7441 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed43/hmc/best.pth |
| HMC | cbramod | seed44 | done | 0.0001 | 32 | 16 | 6 | kappa | 0.7010 | 0.7317 | 0.6784 | 0.7481 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed44/hmc/best.pth |
| HMC | cbramod | seed45 | done | 0.0001 | 32 | 17 | 7 | kappa | 0.6995 | 0.7299 | 0.6774 | 0.7488 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed45/hmc/best.pth |
| HMC | cbramod | seed46 | done | 0.0001 | 32 | 17 | 7 | kappa | 0.6989 | 0.7272 | 0.6733 | 0.7442 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed46/hmc/best.pth |
| Mumtaz2016 | cbramod | seed42 | done | 0.0001 | 32 | 14 | 4 | pr_auc | 1.0000 | 0.9142 | 0.9840 | 0.9827 | 0.9923 | -0.0083 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed42/mumtaz2016/best.pth |
| Mumtaz2016 | cbramod | seed43 | done | 0.0001 | 32 | 15 | 5 | pr_auc | 1.0000 | 0.8939 | 0.9872 | 0.9859 | 0.9923 | -0.0051 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed43/mumtaz2016/best.pth |
| Mumtaz2016 | cbramod | seed44 | done | 0.0001 | 32 | 17 | 7 | pr_auc | 0.9999 | 0.8982 | 0.9777 | 0.9756 | 0.9923 | -0.0146 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed44/mumtaz2016/best.pth |
| Mumtaz2016 | cbramod | seed45 | done | 0.0001 | 32 | 16 | 6 | pr_auc | 0.9999 | 0.8932 | 0.9561 | 0.9491 | 0.9923 | -0.0362 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed45/mumtaz2016/best.pth |
| Mumtaz2016 | cbramod | seed46 | done | 0.0001 | 32 | 14 | 4 | pr_auc | 0.9998 | 0.9044 | 0.9784 | 0.9780 | 0.9923 | -0.0139 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed46/mumtaz2016/best.pth |
| PhysioNet-MI | cbramod | seed42 | done | 0.0001 | 32 | 16 | 6 | kappa | 0.4418 | 0.6318 | 0.5090 | 0.6320 | 0.5222 | -0.0132 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed42/physionet_mi/best.pth |
| PhysioNet-MI | cbramod | seed43 | done | 0.0001 | 32 | 16 | 6 | kappa | 0.4425 | 0.6174 | 0.4898 | 0.6171 | 0.5222 | -0.0324 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed43/physionet_mi/best.pth |
| PhysioNet-MI | cbramod | seed44 | done | 0.0001 | 32 | 16 | 6 | kappa | 0.4372 | 0.6168 | 0.4890 | 0.6165 | 0.5222 | -0.0332 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed44/physionet_mi/best.pth |
| PhysioNet-MI | cbramod | seed45 | done | 0.0001 | 32 | 16 | 6 | kappa | 0.4417 | 0.6124 | 0.4831 | 0.6133 | 0.5222 | -0.0391 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed45/physionet_mi/best.pth |
| PhysioNet-MI | cbramod | seed46 | done | 0.0001 | 32 | 15 | 5 | kappa | 0.4402 | 0.6130 | 0.4839 | 0.6118 | 0.5222 | -0.0383 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed46/physionet_mi/best.pth |
| TUAB | cbramod | seed42 | done | 0.0001 | 32 | 10 | 1 | pr_auc | 0.9271 | 0.8070 | 0.8952 | 0.8910 | 0.9221 | -0.0269 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed42/tuab/best.pth |
| TUAB | cbramod | seed43 | done | 0.0001 | 32 | 10 | 1 | pr_auc | 0.9225 | 0.8022 | 0.8894 | 0.8846 | 0.9221 | -0.0327 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed43/tuab/best.pth |
| TUAB | cbramod | seed44 | done | 0.0001 | 32 | 10 | 1 | pr_auc | 0.9261 | 0.8075 | 0.8906 | 0.8899 | 0.9221 | -0.0315 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed44/tuab/best.pth |
| TUAB | cbramod | seed45 | done | 0.0001 | 32 | 10 | 1 | pr_auc | 0.9221 | 0.8060 | 0.8909 | 0.8854 | 0.9221 | -0.0312 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed45/tuab/best.pth |
| TUAB | cbramod | seed46 | done | 0.0001 | 32 | 10 | 1 | pr_auc | 0.9250 | 0.8104 | 0.8956 | 0.8935 | 0.9221 | -0.0265 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed46/tuab/best.pth |
| FACED | reve | seed42 | done | 0.0001 | 32 | 19 | 9 | kappa | 0.4574 | 0.4838 | 0.4189 | 0.4858 | 0.5080 | -0.0891 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed42/faced/best.pth |
| FACED | reve | seed43 | done | 0.0001 | 32 | 31 | 21 | kappa | 0.5032 | 0.5480 | 0.4898 | 0.5477 | 0.5080 | -0.0182 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed43/faced/best.pth |
| FACED | reve | seed44 | done | 0.0001 | 32 | 14 | 4 | kappa | 0.4785 | 0.4924 | 0.4285 | 0.4955 | 0.5080 | -0.0795 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed44/faced/best.pth |
| FACED | reve | seed45 | done | 0.0001 | 32 | 31 | 21 | kappa | 0.5017 | 0.5356 | 0.4753 | 0.5358 | 0.5080 | -0.0327 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed45/faced/best.pth |
| FACED | reve | seed46 | done | 0.0001 | 32 | 16 | 6 | kappa | 0.4898 | 0.5039 | 0.4401 | 0.5049 | 0.5080 | -0.0679 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed46/faced/best.pth |
| HMC | reve | seed42 | done | 0.0001 | 32 | 14 | 4 | kappa | 0.7098 | 0.7339 | 0.6804 | 0.7497 | 0.6982 | -0.0178 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed42/hmc/best.pth |
| HMC | reve | seed43 | done | 0.0001 | 32 | 14 | 4 | kappa | 0.7114 | 0.7351 | 0.6791 | 0.7481 | 0.6982 | -0.0191 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed43/hmc/best.pth |
| HMC | reve | seed44 | done | 0.0001 | 32 | 14 | 4 | kappa | 0.7097 | 0.7304 | 0.6766 | 0.7461 | 0.6982 | -0.0216 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed44/hmc/best.pth |
| HMC | reve | seed45 | done | 0.0001 | 32 | 14 | 4 | kappa | 0.7104 | 0.7362 | 0.6844 | 0.7517 | 0.6982 | -0.0138 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed45/hmc/best.pth |
| HMC | reve | seed46 | done | 0.0001 | 32 | 13 | 3 | kappa | 0.7096 | 0.7361 | 0.6860 | 0.7536 | 0.6982 | -0.0122 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed46/hmc/best.pth |
| Mumtaz2016 | reve | seed42 | done | 0.0001 | 32 | 13 | 3 | pr_auc | 1.0000 | 0.9515 | 0.9972 | 0.9971 | 0.9961 | 0.0011 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed42/mumtaz2016/best.pth |
| Mumtaz2016 | reve | seed43 | done | 0.0001 | 32 | 13 | 3 | pr_auc | 1.0000 | 0.9332 | 0.9947 | 0.9946 | 0.9961 | -0.0014 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed43/mumtaz2016/best.pth |
| Mumtaz2016 | reve | seed44 | done | 0.0001 | 32 | 27 | 17 | pr_auc | 1.0000 | 0.9124 | 0.9839 | 0.9823 | 0.9961 | -0.0122 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed44/mumtaz2016/best.pth |
| Mumtaz2016 | reve | seed45 | done | 0.0001 | 32 | 13 | 3 | pr_auc | 1.0000 | 0.9114 | 0.9849 | 0.9824 | 0.9961 | -0.0112 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed45/mumtaz2016/best.pth |
| Mumtaz2016 | reve | seed46 | done | 0.0001 | 32 | 12 | 2 | pr_auc | 1.0000 | 0.9181 | 0.9953 | 0.9946 | 0.9961 | -0.0008 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed46/mumtaz2016/best.pth |
| PhysioNet-MI | reve | seed42 | done | 0.0001 | 32 | 26 | 16 | kappa | 0.5056 | 0.6977 | 0.5969 | 0.6981 | 0.5306 | 0.0664 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed42/physionet_mi/best.pth |
| PhysioNet-MI | reve | seed43 | done | 0.0001 | 32 | 14 | 4 | kappa | 0.5071 | 0.6955 | 0.5940 | 0.6957 | 0.5306 | 0.0634 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed43/physionet_mi/best.pth |
| PhysioNet-MI | reve | seed44 | done | 0.0001 | 32 | 30 | 20 | kappa | 0.5033 | 0.6955 | 0.5940 | 0.6964 | 0.5306 | 0.0634 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed44/physionet_mi/best.pth |
| PhysioNet-MI | reve | seed45 | done | 0.0001 | 32 | 20 | 10 | kappa | 0.5079 | 0.6849 | 0.5799 | 0.6849 | 0.5306 | 0.0493 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed45/physionet_mi/best.pth |
| PhysioNet-MI | reve | seed46 | done | 0.0001 | 32 | 16 | 6 | kappa | 0.5071 | 0.6844 | 0.5792 | 0.6835 | 0.5306 | 0.0486 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed46/physionet_mi/best.pth |
| TUAB | reve | seed42 | done | 0.0001 | 32 | 10 | 4 | pr_auc | 0.9213 | 0.7974 | 0.8757 | 0.8739 | 0.9281 | -0.0524 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed42/tuab/best.pth |
| TUAB | reve | seed43 | done | 0.0001 | 32 | 10 | 3 | pr_auc | 0.9193 | 0.8088 | 0.8825 | 0.8820 | 0.9281 | -0.0456 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed43/tuab/best.pth |
| TUAB | reve | seed44 | done | 0.0001 | 32 | 10 | 4 | pr_auc | 0.9219 | 0.8035 | 0.8818 | 0.8785 | 0.9281 | -0.0463 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed44/tuab/best.pth |
| TUAB | reve | seed45 | done | 0.0001 | 32 | 10 | 3 | pr_auc | 0.9176 | 0.8125 | 0.8817 | 0.8830 | 0.9281 | -0.0464 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed45/tuab/best.pth |
| TUAB | reve | seed46 | done | 0.0001 | 32 | 10 | 1 | pr_auc | 0.9228 | 0.8095 | 0.8947 | 0.8875 | 0.9281 | -0.0334 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed46/tuab/best.pth |
