# Phase-Interleaved Folding Results

This file is the source of truth for results produced after correcting the EEG
adapter to phase-interleaved folding:

```text
I[c * P + p, w] = X[c, w * P + p]
```

Results from the superseded contiguous-chunk adapter must not be mixed with the
tables below. All checkpoints are selected on validation metrics and evaluated
once on the test split. Reported standard deviations use the population
definition over seeds 3407, 3408, and 3409.

The consolidated reproduction entrypoint for the finalized recipes is
`experiments/run_finalized_efficientnet_b0_3seed.sh`. Historical search
launchers are retained separately and should not be treated as the canonical
paper runner.

## Canonical reproduction recipes

Common settings are EfficientNet-B0 ImageNet initialization, full-model
fine-tuning, AdamW with cosine decay, dropout .1, float16 AMP, no time-roll or
amplitude-scale augmentation, validation-only checkpoint selection, and one
final test evaluation. The consolidated runner fixes the following per-dataset
overrides for all three reporting seeds:

| Dataset | P | Epochs | Batch | LR | Weight decay | Select | Special setting |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CHB-MIT | 2 | 10 | 32 | 1e-3 | 5e-3 | PR-AUC | none |
| TUAB | 2 | 5 | 32 | 1e-3 | 5e-4 | PR-AUC | grad clip 1 |
| TUEV | 4 | 10 | 32 | 1e-3 | 5e-3 | kappa | none |
| ISRUC | 8 | 50 | 16 | 1e-3 | 5e-3 | kappa | mirror p=.5 |
| FACED | 2 | 50 | 32 | 1e-3 | 5e-3 | kappa | none |
| SEED-V | 8 | 50 | 32 | 5e-4 | 5e-3 | kappa | none |
| PhysioNet-MI | 1 | 30 | 32 | 2e-3 | 5e-3 | kappa | no folding |
| SHU-MI | 4 | 20 | 32 | 1e-3 | 5e-3 | PR-AUC | clip 512, divide 64 |
| BCIC2020-3 | 1 | 30 | 32 | 1e-3 | 5e-3 | kappa | no folding |
| Mumtaz2016 | 2 | 30 | 32 | 5e-4 | 5e-2 | PR-AUC | none |
| MentalArithmetic | 2 | 10 | 32 | 5e-4 | 1e-2 | PR-AUC | none |

## Finalized three-seed results

Binary datasets report PR-AUC/AUROC. Multiclass datasets report Cohen's
kappa/weighted F1.

| Dataset | Seed 3407 | Seed 3408 | Seed 3409 | Mean +/- std | Published CBraMod |
| --- | --- | --- | --- | --- | --- |
| BCIC2020-3 | .52500/.62020 | .52500/.62161 | .49167/.59398 | **.51389 +/- .01571 / .61193 +/- .01271** | .4216/.5383 |
| Mumtaz2016 | .97603/.97586 | .97342/.96885 | .97167/.96877 | .97371 +/- .00179 / .97116 +/- .00332 | **.9923/.9921** |
| MentalArithmetic | .73476/.87630 | .57663/.80266 | .64232/.83087 | **.65124 +/- .06486 / .83661 +/- .03034** | .6267/.7905 |

Primary-metric differences versus the published CBraMod references:

| Dataset | Difference (percentage points) | Reading |
| --- | ---: | --- |
| BCIC2020-3 | +9.23 kappa | stable improvement |
| Mumtaz2016 | -1.86 PR-AUC | stable deficit near the performance ceiling |
| MentalArithmetic | +2.45 PR-AUC | mean improvement with substantial seed variance |

## Selected training configurations

Common settings: EfficientNet-B0 ImageNet pretrained weights, joint full-model
fine-tuning, AdamW, cosine LR decay, batch size 32, dropout .1, AMP float16, no
data augmentation, no backbone/head LR split, validation checkpoint selection,
and one final test evaluation.

| Dataset | Folding P | Epochs | LR | Weight decay | Selection metric | Input after folding (before padding) |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| BCIC2020-3 | 1 | 30 | 1e-3 | 5e-3 | kappa | 64 x 600 |
| Mumtaz2016 | 2 | 30 | 5e-4 | 5e-2 | PR-AUC | 38 x 500 |
| MentalArithmetic | 2 | 10 | 5e-4 | 1e-2 | PR-AUC | 40 x 500 |

## Provenance

- Initial P=2/P=4 pilot:
  `experiments/logs/phase_fold_small_pilot_v1/launcher.log`
- Hyperparameter and geometry search:
  `experiments/logs/phase_fold_small_followup_v1/launcher.log`
- Seed-completion runs:
  `experiments/logs/phase_fold_best_3seed_v1/launcher.log`
- Checkpoints are under the matching experiment names in
  `experiments/checkpoints/`.

The BCIC2020-3 and Mumtaz2016 configurations were selected using exploratory
seed-3407 searches. A publication-grade confirmatory study should freeze these
configurations before evaluating a new split or an untouched set of seeds.

## Finalized three-seed results: remaining five datasets

The seed-3407 values for ISRUC, SEED-V, CHB-MIT, and TUAB were produced by the
completed pilot (`phase_fold_remaining5_pilot_v1`) using the same folding
factors, optimizer, learning rates, weight decay, augmentation settings, and
epoch budgets as the seed-3408/3409 runs in `phase_fold_remaining5_3seed_v1`.

TUEV was re-run and re-finalized after this table was first written. Its three
seeds now come from `tuev_p4_lr1e3_10ep_3seed_v1` (P=4, lr=1e-3, wd=5e-3,
10 epochs, kappa selection, EfficientNet-B0 ImageNet weights), replacing the
earlier P=2 / lr=3e-4 / 50-epoch configuration that scored .66650 kappa.

| Dataset | Seed 3407 | Seed 3408 | Seed 3409 | Mean +/- std | Published CBraMod |
| --- | --- | --- | --- | --- | --- |
| TUEV | .68710/.83240 | .75656/.87256 | .70530/.84046 | **.71632 +/- .02941 / .84847 +/- .01735** | .6744/.8331 |
| ISRUC | .76239/.81730 | .77839/.82397 | .75022/.79910 | **.76367 +/- .01154 / .81346 +/- .01051** | .7442/.8011 |
| SEED-V | .24491/.40159 | .26001/.41536 | .26468/.41578 | **.25653 +/- .00844 / .41091 +/- .00659** | .2569/.4101 |
| CHB-MIT | .48090/.90161 | .42558/.91119 | .42145/.92480 | **.44264 +/- .02710 / .91253 +/- .00951** | .3689/.8892 |
| TUAB | .88650/.87779 | .88581/.87940 | .88152/.87751 | **.88461 +/- .00220 / .87823 +/- .00083** | .9221/.9156 |

Binary datasets report PR-AUC/AUROC; multiclass datasets report Cohen's
kappa/weighted F1. Standard deviations use the population definition over
seeds 3407, 3408, and 3409.

## Medium-scale seed-3407 pilots

These are exploratory single-seed results and are not yet finalized. FACED and
PhysioNet-MI report kappa/weighted F1; SHU-MI reports PR-AUC/AUROC.

| Dataset | P=1 or P=2 result | Alternative P result | Selected pilot | Published CBraMod | Primary delta |
| --- | --- | --- | --- | --- | ---: |
| FACED | P=1: .35385/.43628 | P=2: .39297/.47101 | P=2 | .5041/.5618 | -11.11 kappa points |
| PhysioNet-MI | P=1: .52444/.64685 | P=2: .48968/.61522 | P=1 | .5222/.6427 | +.22 kappa points |
| SHU-MI | P=2: .69028/.67533 | P=4: .65808/.64384 | P=4, scale 64 (see below) | .7139/.6988 | -3.03 PR-AUC |

Selected pilot configurations:

| Dataset | P | Epochs | LR | Weight decay | Augmentation | Selection metric |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| FACED | 2 | 50 | 1e-3 | 5e-3 | none | kappa |
| PhysioNet-MI | 1 | 30 | 2e-3 | 5e-3 | none | kappa |
| SHU-MI | 4 | 20 | 1e-3 | 5e-3 | none, scale 64 | PR-AUC |

Pilot log: `experiments/logs/phase_fold_medium_pilot_v1/launcher.log`.

The first SHU-MI pilot accidentally used `LogUniform(.25,1.25)`. Its geometric
center is .559, so it systematically favors amplitude shrinkage. Those values
are retained for provenance but must not be treated as the selected SHU-MI
configuration. The correction reruns `LogUniform(.75,1.25)` plus a no-scale
control.

## Finalized medium-scale results

| Dataset | Seed 3407 | Seed 3408 | Seed 3409 | Mean +/- std | Published CBraMod |
| --- | --- | --- | --- | --- | --- |
| FACED, P=2 | .39297/.47101 | .38057/.45414 | .32736/.40821 | .36697 +/- .02846 / .44445 +/- .02654 | **.5041/.5618** |
| PhysioNet-MI, P=1 | .52444/.64685 | .53637/.65420 | .50598/.63330 | **.52226 +/- .01250 / .64478 +/- .00866** | .5222/.6427 |
| SHU-MI, P=4 | .71226/.70032 | .69027/.68997 | .64812/.63658 | .68355 +/- .02661 / .67562 +/- .02793 | **.7139/.6988** |

FACED and PhysioNet-MI report kappa/weighted F1; SHU-MI reports PR-AUC/AUROC.
PhysioNet-MI is at parity in kappa (+.006 percentage points) and improves F1 by
.21 points. FACED remains 13.71 kappa points below the published reference.
SHU-MI (P=4, `--shu_scale 64`) sits 3.03 PR-AUC points below the reference.

The corrected SHU-MI seed-3407 augmentation controls are:

| Setting | PR-AUC/AUROC |
| --- | --- |
| P=2, no amplitude scaling | .66978/.67159 |
| P=2, LogUniform(.75,1.25), p=.5 | .66800/.64648 |
| P=4, LogUniform(.75,1.25), p=.5 | .67944/.66855 |

Amplitude scaling did not improve the matched P=2 control.

## SHU-MI input-scale fix

SHU-MI's loader originally clipped to `[-512, 512]` with **no divisor**, unlike
every other dataset's `clip[-1024,1024] / 32`. Its raw values (std ~4.7, mostly
within +/-50) reached the pretrained backbone ~16x larger than the other
datasets' scaled inputs. A `--shu_scale` divisor sweep on seed 3407 showed:

| scale | PR-AUC/AUROC |
| ---: | --- |
| 1 (no divisor) | .67960/.64684 |
| 4 | .69326/.68434 |
| 8 | .67609/.65996 |
| 16 | .68450/.68804 |
| 32 | .69593/.68549 |
| 64 | .71226/.70032 |
| 128 | .69776/.69716 |
| 256 | .70045/.69285 |

The peak is `--shu_scale 64`; the 3-seed result is `.68355 +/- .02661 /
.67562 +/- .02793` (PR-AUC/AUROC), up from the no-divisor `.66957/.65612`
(+2.0 pp AUROC). An lr probe at scale 64 (5e-4 / 1e-3 / 2e-3) confirmed
`lr=1e-3` remains optimal. The loader now divides by `--shu_scale` (default 64).

Logs:

- `experiments/logs/phase_fold_medium_3seed_v1/launcher.log`
- `experiments/logs/shu_amplitude_correction_v1/launcher.log`
- `experiments/logs/shu_scale_sweep_v1/` (scale 4/8/16/32/64/128/256)
- `experiments/logs/shu_p4_scale64_3seed_v1/` (finalized 3-seed)

## Recent exploration (not finalized)

Lowering lr to 1e-4 dramatically stabilizes MentalArithmetic at the cost of a
lower mean; P=4 and bf16 do not help PhysioNet-MI; SEED-V P=1 is worse than P=8.

| Experiment | Result (mean +/- std) | Verdict |
| --- | --- | --- |
| MentalArithmetic P=2 lr=5e-4 (finalized) | .65124 +/- .06486 / .83661 +/- .03034 | high mean, high variance |
| MentalArithmetic P=2 lr=1e-4 | .60255 +/- .03071 / .80208 +/- .01577 | stable, lower mean |
| MentalArithmetic P=4 (lr 1e-4/1e-3, bf16) | PR-AUC ~.41-.57 | worse than P=2 |
| PhysioNet-MI P=2 lr=1e-3 | .47519 +/- .00800 / .60641 +/- .00602 | worse than P=1 |
| PhysioNet-MI P=1 bf16 | .51164 +/- .00757 / .63489 +/- .00342 | slightly worse than float16 |
| SEED-V P=1 | .21091 +/- .00764 / .37358 +/- .00644 | worse than P=8 (.25653) |
| TUEV P=4 lr=1e-3 | .71632 +/- .02941 / .84847 +/- .01735 | **adopted as finalized** |
