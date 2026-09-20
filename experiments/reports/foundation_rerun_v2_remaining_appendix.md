# Foundation-model matched-pipeline rerun (CBraMod / REVE)

Protocol: the v2 unified recipe (learning rate 1e-4, warmup 3 epochs at factor 0.1, EMA 0.995, weight decay 5e-4, gradient clipping 1.0, early stop 10) with each dataset's epochs and selection metric from `configs/downstream.py`. Inputs are normalized by the dataset loaders with each model's released convention (CBraMod microvolt / 100; REVE per-dataset `scale_factor`; no clipping). Checkpoints are selected on the validation metric and evaluated once on test per seed (seeds 42-46).

Models: CBraMod (4.92M parameters, full fine-tuning of the released checkpoint) and REVE-Base (69.19M parameters, full fine-tuning). No encoder is frozen.

### Deviations and provenance

- Training recipe: the v2 unified recipe is applied to both models; the learning rate (1e-4) remains the only per-model adaptation.
- CBraMod: the released recipe uses a multi-LR setting (encoder 1e-4, head 1e-3*sqrt(batch/256)); this rerun uses one uniform learning rate. AdamW in both cases.
- REVE: the released procedure warm-starts with a linear probe before full fine-tuning, applies mixup and StableAdamW, and the paper additionally describes LoRA and model souping; this rerun is a single fine-tuning stage per seed with AdamW and no mixup/LoRA/souping.
- Inputs: identical pre-processed benchmark arrays (inherited filters and splits); each loader applies the model's released normalization (CBraMod microvolt/100; REVE per-dataset scale_factor; no clipping). REVE's released TUAB pipeline uses a 21-channel memmap; this rerun feeds it the same 16-channel bipolar TUAB arrays as CBraMod.
- Protocol: validation-selected checkpoint, one final test per seed, seeds 42-46, population std.
- REVE fallback specs: CHB-MIT, SEED-V, SHU-MI are not part of the released REVE benchmark, so their REVE runs use microvolt / 100, pooling=last and dropout 0.5 and are reported as reference-only. SEED-V maps CB1/CB2 to the inferior occipital OI1h/OI2h positions because the released position bank has no cerebellar entries.
- REVE ISRUC runs at batch 8 instead of the configured batch size 16 because the 22-layer encoder exceeds the GPU memory at batch 16 (CUDA OOM); this is the only batch-size deviation.

## Suggested main-text sentence

> To assess the comparability of published foundation-model references, we additionally reran CBraMod and REVE on seven datasets under the inherited partitions and final-test protocol. The rerun results and deviations from published values are reported in Appendix X.

## Summary (completed seeds, mean +/- population standard deviation)

| Dataset | Model | Seeds | Rerun BA | Rerun primary | Published | Difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BCIC2020-3 | cbramod | 5 | 0.5952 +/- 0.0103 | 0.4940 +/- 0.0129 (kappa) | 0.4216 +/- 0.0163 | 0.0724 |
| BCIC2020-3 | reve | 5 | 0.2856 +/- 0.0971 | 0.1070 +/- 0.1213 (kappa) | 0.4543 +/- 0.0154 | -0.3473 |
| CHB-MIT | cbramod | 5 | 0.7418 +/- 0.0197 | 0.3686 +/- 0.0521 (pr_auc) | 0.3689 +/- 0.0382 | -0.0003 |
| CHB-MIT | reve | 5 | 0.7316 +/- 0.0307 | 0.4623 +/- 0.0264 (pr_auc) | -- | -- |
| ISRUC | cbramod | 5 | 0.7671 +/- 0.0085 | 0.7390 +/- 0.0100 (kappa) | 0.7442 +/- 0.0152 | -0.0052 |
| ISRUC | reve | 5 | 0.8012 +/- 0.0037 | 0.7766 +/- 0.0046 (kappa) | 0.7500 +/- 0.0156 | 0.0266 |
| MentalArithmetic | cbramod | 5 | 0.7451 +/- 0.0232 | 0.5180 +/- 0.0370 (pr_auc) | 0.6267 +/- 0.0099 | -0.1087 |
| MentalArithmetic | reve | 5 | 0.6736 +/- 0.0412 | 0.6679 +/- 0.0822 (pr_auc) | 0.7470 +/- 0.0807 | -0.0791 |
| SEED-V | cbramod | 5 | 0.3812 +/- 0.0033 | 0.2312 +/- 0.0045 (kappa) | 0.2569 +/- 0.0143 | -0.0257 |
| SEED-V | reve | 5 | 0.2617 +/- 0.0614 | 0.0784 +/- 0.0770 (kappa) | -- | -- |
| SHU-MI | cbramod | 5 | 0.6242 +/- 0.0050 | 0.7021 +/- 0.0103 (pr_auc) | 0.7139 +/- 0.0088 | -0.0118 |
| SHU-MI | reve | 5 | 0.6356 +/- 0.0047 | 0.7187 +/- 0.0101 (pr_auc) | -- | -- |
| TUEV | cbramod | 5 | 0.5030 +/- 0.0178 | 0.5694 +/- 0.0217 (kappa) | 0.6744 +/- 0.0121 | -0.1050 |
| TUEV | reve | 5 | 0.6587 +/- 0.0276 | 0.6724 +/- 0.0215 (kappa) | 0.6783 +/- 0.0199 | -0.0059 |

## Per-run detail

| Dataset | Model | Seed | Status | LR | Batch | Epochs | Best epoch | Val sel | Val best | Test BA | Test primary | Test secondary | Published | Difference | Checkpoint |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BCIC2020-3 | cbramod | seed42 | done | 0.0001 | 32 | 30 | 27 | kappa | 0.4933 | 0.5880 | 0.4850 | 0.5879 | 0.4216 | 0.0634 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed42/bcic2020_3/best.pth |
| BCIC2020-3 | cbramod | seed43 | done | 0.0001 | 32 | 30 | 30 | kappa | 0.4950 | 0.5853 | 0.4817 | 0.5845 | 0.4216 | 0.0601 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed43/bcic2020_3/best.pth |
| BCIC2020-3 | cbramod | seed44 | done | 0.0001 | 32 | 30 | 27 | kappa | 0.5167 | 0.6000 | 0.5000 | 0.5997 | 0.4216 | 0.0784 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed44/bcic2020_3/best.pth |
| BCIC2020-3 | cbramod | seed45 | done | 0.0001 | 32 | 30 | 28 | kappa | 0.5117 | 0.5893 | 0.4867 | 0.5891 | 0.4216 | 0.0651 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed45/bcic2020_3/best.pth |
| BCIC2020-3 | cbramod | seed46 | done | 0.0001 | 32 | 30 | 29 | kappa | 0.4933 | 0.6133 | 0.5167 | 0.6130 | 0.4216 | 0.0951 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed46/bcic2020_3/best.pth |
| CHB-MIT | cbramod | seed42 | done | 0.0001 | 32 | 10 | 8 | pr_auc | 0.4450 | 0.7170 | 0.3498 | 0.8950 | 0.3689 | -0.0191 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed42/chb_mit/best.pth |
| CHB-MIT | cbramod | seed43 | done | 0.0001 | 32 | 10 | 10 | pr_auc | 0.4043 | 0.7331 | 0.4229 | 0.9033 | 0.3689 | 0.0540 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed43/chb_mit/best.pth |
| CHB-MIT | cbramod | seed44 | done | 0.0001 | 32 | 10 | 8 | pr_auc | 0.4696 | 0.7392 | 0.3141 | 0.8997 | 0.3689 | -0.0548 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed44/chb_mit/best.pth |
| CHB-MIT | cbramod | seed45 | done | 0.0001 | 32 | 10 | 8 | pr_auc | 0.4700 | 0.7429 | 0.3182 | 0.8968 | 0.3689 | -0.0507 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed45/chb_mit/best.pth |
| CHB-MIT | cbramod | seed46 | done | 0.0001 | 32 | 10 | 10 | pr_auc | 0.4716 | 0.7770 | 0.4378 | 0.9130 | 0.3689 | 0.0689 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed46/chb_mit/best.pth |
| ISRUC | cbramod | seed42 | done | 0.0001 | 16 | 15 | 10 | kappa | 0.7272 | 0.7778 | 0.7499 | 0.8049 | 0.7442 | 0.0057 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed42/isruc/best.pth |
| ISRUC | cbramod | seed43 | done | 0.0001 | 16 | 15 | 8 | kappa | 0.7168 | 0.7633 | 0.7349 | 0.7912 | 0.7442 | -0.0093 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed43/isruc/best.pth |
| ISRUC | cbramod | seed44 | done | 0.0001 | 16 | 15 | 13 | kappa | 0.7244 | 0.7717 | 0.7412 | 0.7993 | 0.7442 | -0.0030 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed44/isruc/best.pth |
| ISRUC | cbramod | seed45 | done | 0.0001 | 16 | 15 | 7 | kappa | 0.7217 | 0.7529 | 0.7218 | 0.7813 | 0.7442 | -0.0224 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed45/isruc/best.pth |
| ISRUC | cbramod | seed46 | done | 0.0001 | 16 | 15 | 10 | kappa | 0.7233 | 0.7700 | 0.7472 | 0.8011 | 0.7442 | 0.0030 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed46/isruc/best.pth |
| MentalArithmetic | cbramod | seed42 | done | 0.0001 | 64 | 30 | 23 | pr_auc | 0.8257 | 0.7569 | 0.5552 | 0.8582 | 0.6267 | -0.0715 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed42/mentalarithmetic/best.pth |
| MentalArithmetic | cbramod | seed43 | done | 0.0001 | 64 | 30 | 29 | pr_auc | 0.8235 | 0.7014 | 0.4468 | 0.7595 | 0.6267 | -0.1799 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed43/mentalarithmetic/best.pth |
| MentalArithmetic | cbramod | seed44 | done | 0.0001 | 64 | 24 | 14 | pr_auc | 0.8272 | 0.7431 | 0.5325 | 0.8274 | 0.6267 | -0.0942 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed44/mentalarithmetic/best.pth |
| MentalArithmetic | cbramod | seed45 | done | 0.0001 | 64 | 30 | 21 | pr_auc | 0.8227 | 0.7674 | 0.5274 | 0.8317 | 0.6267 | -0.0993 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed45/mentalarithmetic/best.pth |
| MentalArithmetic | cbramod | seed46 | done | 0.0001 | 64 | 24 | 14 | pr_auc | 0.8241 | 0.7569 | 0.5282 | 0.8367 | 0.6267 | -0.0986 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed46/mentalarithmetic/best.pth |
| SEED-V | cbramod | seed42 | done | 0.0001 | 32 | 30 | 20 | kappa | 0.2042 | 0.3833 | 0.2354 | 0.3923 | 0.2569 | -0.0215 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed42/seed_v/best.pth |
| SEED-V | cbramod | seed43 | done | 0.0001 | 32 | 20 | 10 | kappa | 0.2095 | 0.3853 | 0.2358 | 0.3929 | 0.2569 | -0.0211 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed43/seed_v/best.pth |
| SEED-V | cbramod | seed44 | done | 0.0001 | 32 | 20 | 10 | kappa | 0.2060 | 0.3782 | 0.2260 | 0.3861 | 0.2569 | -0.0309 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed44/seed_v/best.pth |
| SEED-V | cbramod | seed45 | done | 0.0001 | 32 | 16 | 6 | kappa | 0.2083 | 0.3827 | 0.2333 | 0.3896 | 0.2569 | -0.0236 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed45/seed_v/best.pth |
| SEED-V | cbramod | seed46 | done | 0.0001 | 32 | 18 | 8 | kappa | 0.2046 | 0.3765 | 0.2258 | 0.3839 | 0.2569 | -0.0311 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed46/seed_v/best.pth |
| SHU-MI | cbramod | seed42 | done | 0.0001 | 32 | 10 | 5 | pr_auc | 0.7736 | 0.6279 | 0.7139 | 0.7016 | 0.7139 | -0.0000 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed42/shu_mi/best.pth |
| SHU-MI | cbramod | seed43 | done | 0.0001 | 32 | 10 | 6 | pr_auc | 0.7712 | 0.6300 | 0.7029 | 0.6869 | 0.7139 | -0.0110 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed43/shu_mi/best.pth |
| SHU-MI | cbramod | seed44 | done | 0.0001 | 32 | 10 | 7 | pr_auc | 0.7739 | 0.6228 | 0.6997 | 0.6872 | 0.7139 | -0.0142 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed44/shu_mi/best.pth |
| SHU-MI | cbramod | seed45 | done | 0.0001 | 32 | 10 | 5 | pr_auc | 0.7747 | 0.6246 | 0.7101 | 0.6962 | 0.7139 | -0.0038 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed45/shu_mi/best.pth |
| SHU-MI | cbramod | seed46 | done | 0.0001 | 32 | 10 | 7 | pr_auc | 0.7746 | 0.6156 | 0.6841 | 0.6657 | 0.7139 | -0.0298 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed46/shu_mi/best.pth |
| TUEV | cbramod | seed42 | done | 0.0001 | 32 | 10 | 1 | kappa | 0.5214 | 0.5314 | 0.6092 | 0.7943 | 0.6744 | -0.0652 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed42/tuev/best.pth |
| TUEV | cbramod | seed43 | done | 0.0001 | 32 | 10 | 1 | kappa | 0.4919 | 0.4854 | 0.5448 | 0.7623 | 0.6744 | -0.1296 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed43/tuev/best.pth |
| TUEV | cbramod | seed44 | done | 0.0001 | 32 | 10 | 1 | kappa | 0.5121 | 0.5158 | 0.5719 | 0.7749 | 0.6744 | -0.1025 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed44/tuev/best.pth |
| TUEV | cbramod | seed45 | done | 0.0001 | 32 | 10 | 1 | kappa | 0.5203 | 0.4956 | 0.5603 | 0.7693 | 0.6744 | -0.1141 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed45/tuev/best.pth |
| TUEV | cbramod | seed46 | done | 0.0001 | 32 | 10 | 1 | kappa | 0.5192 | 0.4870 | 0.5607 | 0.7715 | 0.6744 | -0.1137 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/cbramod/seed46/tuev/best.pth |
| BCIC2020-3 | reve | seed42 | done | 0.0001 | 32 | 12 | 2 | kappa | 0.0283 | 0.2173 | 0.0217 | 0.1570 | 0.4543 | -0.4326 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed42/bcic2020_3/best.pth |
| BCIC2020-3 | reve | seed43 | done | 0.0001 | 32 | 30 | 27 | kappa | 0.3817 | 0.4720 | 0.3400 | 0.4723 | 0.4543 | -0.1143 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed43/bcic2020_3/best.pth |
| BCIC2020-3 | reve | seed44 | done | 0.0001 | 32 | 30 | 30 | kappa | 0.1133 | 0.2880 | 0.1100 | 0.2844 | 0.4543 | -0.3443 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed44/bcic2020_3/best.pth |
| BCIC2020-3 | reve | seed45 | done | 0.0001 | 32 | 11 | 1 | kappa | 0.0350 | 0.2107 | 0.0133 | 0.1190 | 0.4543 | -0.4410 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed45/bcic2020_3/best.pth |
| BCIC2020-3 | reve | seed46 | done | 0.0001 | 32 | 16 | 6 | kappa | 0.0333 | 0.2400 | 0.0500 | 0.2194 | 0.4543 | -0.4043 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed46/bcic2020_3/best.pth |
| CHB-MIT | reve | seed42 | done | 0.0001 | 32 | 10 | 2 | pr_auc | 0.5684 | 0.6885 | 0.4266 | 0.9087 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed42/chb_mit/best.pth |
| CHB-MIT | reve | seed43 | done | 0.0001 | 32 | 10 | 1 | pr_auc | 0.6262 | 0.7415 | 0.4404 | 0.9066 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed43/chb_mit/best.pth |
| CHB-MIT | reve | seed44 | done | 0.0001 | 32 | 10 | 1 | pr_auc | 0.5652 | 0.7616 | 0.4718 | 0.9013 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed44/chb_mit/best.pth |
| CHB-MIT | reve | seed45 | done | 0.0001 | 32 | 10 | 2 | pr_auc | 0.5477 | 0.7026 | 0.4704 | 0.8916 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed45/chb_mit/best.pth |
| CHB-MIT | reve | seed46 | done | 0.0001 | 32 | 10 | 2 | pr_auc | 0.5499 | 0.7635 | 0.5022 | 0.9232 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed46/chb_mit/best.pth |
| ISRUC | reve | seed42 | done | 0.0001 | 8 | 15 | 6 | kappa | 0.7679 | 0.8035 | 0.7811 | 0.8282 | 0.7500 | 0.0311 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed42/isruc/best.pth |
| ISRUC | reve | seed43 | done | 0.0001 | 8 | 15 | 7 | kappa | 0.7729 | 0.8020 | 0.7785 | 0.8255 | 0.7500 | 0.0285 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed43/isruc/best.pth |
| ISRUC | reve | seed44 | done | 0.0001 | 8 | 15 | 6 | kappa | 0.7691 | 0.7943 | 0.7685 | 0.8169 | 0.7500 | 0.0185 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed44/isruc/best.pth |
| ISRUC | reve | seed45 | done | 0.0001 | 8 | 15 | 7 | kappa | 0.7665 | 0.8013 | 0.7745 | 0.8225 | 0.7500 | 0.0245 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed45/isruc/best.pth |
| ISRUC | reve | seed46 | done | 0.0001 | 8 | 15 | 6 | kappa | 0.7624 | 0.8051 | 0.7803 | 0.8273 | 0.7500 | 0.0303 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed46/isruc/best.pth |
| MentalArithmetic | reve | seed42 | done | 0.0001 | 64 | 30 | 25 | pr_auc | 0.6526 | 0.6632 | 0.6622 | 0.8566 | 0.7470 | -0.0848 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed42/mentalarithmetic/best.pth |
| MentalArithmetic | reve | seed43 | done | 0.0001 | 64 | 30 | 24 | pr_auc | 0.6520 | 0.6354 | 0.6628 | 0.8252 | 0.7470 | -0.0842 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed43/mentalarithmetic/best.pth |
| MentalArithmetic | reve | seed44 | done | 0.0001 | 64 | 30 | 30 | pr_auc | 0.6148 | 0.6632 | 0.5792 | 0.6101 | 0.7470 | -0.1678 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed44/mentalarithmetic/best.pth |
| MentalArithmetic | reve | seed45 | done | 0.0001 | 64 | 30 | 28 | pr_auc | 0.6820 | 0.7535 | 0.8200 | 0.9077 | 0.7470 | 0.0730 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed45/mentalarithmetic/best.pth |
| MentalArithmetic | reve | seed46 | done | 0.0001 | 64 | 23 | 13 | pr_auc | 0.5399 | 0.6528 | 0.6155 | 0.7486 | 0.7470 | -0.1315 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed46/mentalarithmetic/best.pth |
| SEED-V | reve | seed42 | done | 0.0001 | 32 | 50 | 48 | kappa | 0.0123 | 0.2061 | 0.0106 | 0.1495 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed42/seed_v/best.pth |
| SEED-V | reve | seed43 | done | 0.0001 | 32 | 25 | 15 | kappa | 0.1857 | 0.3698 | 0.2146 | 0.3775 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed43/seed_v/best.pth |
| SEED-V | reve | seed44 | done | 0.0001 | 32 | 35 | 25 | kappa | 0.0969 | 0.2795 | 0.0996 | 0.2884 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed44/seed_v/best.pth |
| SEED-V | reve | seed45 | done | 0.0001 | 32 | 14 | 4 | kappa | 0.0762 | 0.2522 | 0.0652 | 0.2333 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed45/seed_v/best.pth |
| SEED-V | reve | seed46 | done | 0.0001 | 32 | 11 | 1 | kappa | 0.0038 | 0.2010 | 0.0018 | 0.0992 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed46/seed_v/best.pth |
| SHU-MI | reve | seed42 | done | 0.0001 | 32 | 10 | 3 | pr_auc | 0.7695 | 0.6405 | 0.7268 | 0.7280 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed42/shu_mi/best.pth |
| SHU-MI | reve | seed43 | done | 0.0001 | 32 | 10 | 6 | pr_auc | 0.7538 | 0.6421 | 0.7181 | 0.7143 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed43/shu_mi/best.pth |
| SHU-MI | reve | seed44 | done | 0.0001 | 32 | 10 | 6 | pr_auc | 0.7584 | 0.6325 | 0.7032 | 0.7029 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed44/shu_mi/best.pth |
| SHU-MI | reve | seed45 | done | 0.0001 | 32 | 10 | 3 | pr_auc | 0.7653 | 0.6307 | 0.7320 | 0.7257 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed45/shu_mi/best.pth |
| SHU-MI | reve | seed46 | done | 0.0001 | 32 | 10 | 5 | pr_auc | 0.7861 | 0.6320 | 0.7135 | 0.7086 | -- | -- | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed46/shu_mi/best.pth |
| TUEV | reve | seed42 | done | 0.0001 | 32 | 10 | 1 | kappa | 0.5411 | 0.6210 | 0.6620 | 0.8245 | 0.6783 | -0.0163 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed42/tuev/best.pth |
| TUEV | reve | seed43 | done | 0.0001 | 32 | 10 | 3 | kappa | 0.5660 | 0.6415 | 0.6418 | 0.8152 | 0.6783 | -0.0365 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed43/tuev/best.pth |
| TUEV | reve | seed44 | done | 0.0001 | 32 | 10 | 4 | kappa | 0.5290 | 0.7034 | 0.7037 | 0.8475 | 0.6783 | 0.0254 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed44/tuev/best.pth |
| TUEV | reve | seed45 | done | 0.0001 | 32 | 10 | 2 | kappa | 0.5178 | 0.6690 | 0.6885 | 0.8386 | 0.6783 | 0.0102 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed45/tuev/best.pth |
| TUEV | reve | seed46 | done | 0.0001 | 32 | 10 | 6 | kappa | 0.5897 | 0.6586 | 0.6659 | 0.8296 | 0.6783 | -0.0124 | experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1/reve/seed46/tuev/best.pth |
