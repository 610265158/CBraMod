# Phase-Interleaved Folding Results

This file is the source of truth for formal results after the EEG adapter was
corrected to phase-interleaved folding:

```text
I[c * P + p, w] = X[c, w * P + p]
```

Every formal checkpoint is selected using validation data and evaluated once
on test. Standard deviations use the population definition. ISRUC is reported
with its locked five-seed recipe (42--46); datasets awaiting five-seed
replacement retain the canonical 3407--3409 sweep. Any `P>1` checkpoint produced by the superseded
contiguous-chunk adapter is stale and must not be quoted.

## Canonical 11-dataset reproduction

The current headline experiment is the **unified all-BF16 min-64 three-seed
reproduction**:

- experiment: `all11_min64_repro_3seed_v1`
- launcher: `experiments/run_all11_min64_repro_3seed.sh`
- compatibility entrypoint: `experiments/run_finalized_efficientnet_b0_3seed.sh`
- launcher log: `experiments/logs/all11_min64_repro_3seed_v1_launcher.log`
- run logs: `experiments/logs/all11_min64_repro_3seed_v1/`
- completed: 2026-08-28 21:41 UTC
- finalized ISRUC experiment: `isruc_p12_bottomrightpad_headstd002_5seed_v1`
- finalized ISRUC launcher: `experiments/run_isruc_bottomrightpad_headstd002_5seed.sh`

The model is ImageNet-pretrained EfficientNet-B0 with full-model fine-tuning.
The original 33-run sweep uses BF16 AMP, validation-only checkpoint selection,
and one final test evaluation. ISRUC is superseded by the dedicated five-seed
run documented below. The fold factor follows the user-specified geometry rule:
choose the smallest valid factor that brings the folded channel dimension to
at least 64 rows; a native channel count already sufficiently close to 64 is
left unfolded (SEED-V has 62 channels and therefore uses `P=1`).

This reproduction changes both folding geometry and numerical precision
relative to several earlier recipes. It is therefore not a clean single-factor
ablation of `P`; it is the new unified reproduction baseline.

## Canonical recipes and shapes

Common settings are EfficientNet-B0 ImageNet initialization, AdamW, cosine
decay to `1e-6`, dropout `.1`, no multi-LR split, and BF16 AMP. Dataset-specific
settings below match the completed launcher.

| Dataset | Input | P | Folded | Padded | Epochs | Batch | LR | WD | Select | Special setting |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| CHB-MIT | `16x2000` | 4 | `64x500` | `64x512` | 10 | 32 | 1e-3 | 5e-3 | PR-AUC | none |
| TUAB | `16x2000` | 4 | `64x500` | `64x512` | 5 | 32 | 1e-3 | 5e-4 | PR-AUC | grad clip 1 |
| TUEV | `16x1000` | 4 | `64x250` | `64x256` | 10 | 32 | 1e-3 | 5e-3 | kappa | none |
| ISRUC | `20x6x6000` | 12 | `72x500` | `96x512` | 15 | 16 | 1e-3 | 5e-3 | kappa | warmup 3, EMA .995, roll+mirror, head std .002 |
| FACED | `32x2000` | 2 | `64x1000` | `64x1024` | 50 | 32 | 1e-3 | 5e-3 | kappa | none |
| SEED-V | `62x200` | 1 | `62x200` | `64x224` | 50 | 32 | 5e-4 | 5e-3 | kappa | no folding |
| PhysioNet-MI | `64x800` | 1 | `64x800` | `64x800` | 30 | 32 | 2e-3 | 5e-3 | kappa | warmup 3, EMA .995 |
| SHU-MI | `32x800` | 2 | `64x400` | `64x416` | 20 | 32 | 1e-3 | 5e-4 | PR-AUC | warmup 3, EMA .995, roll, scale 64 |
| BCIC2020-3 | `64x600` | 1 | `64x600` | `64x608` | 30 | 32 | 1e-3 | 5e-3 | kappa | warmup 3, EMA .995, clip 1, roll |
| Mumtaz2016 | `19x1000` | 4 | `76x250` | `96x256` | 30 | 32 | 1e-3 | 5e-4 | PR-AUC | warmup 3, EMA .995, roll |
| MentalArithmetic | `20x1000` | 4 | `80x250` | `96x256` | 30 | 64 | 1e-3 | 5e-4 | PR-AUC | warmup 3, EMA .99, pos-weight 3, roll+mirror |

## Canonical per-seed results

Binary datasets report PR-AUC/AUROC. Multiclass datasets report Cohen's
kappa/weighted F1. ISRUC is omitted from this three-seed table because its
formal row is replaced by the locked five-seed result immediately below.

| Dataset | Seed 3407 | Seed 3408 | Seed 3409 | Mean +/- population std |
| --- | --- | --- | --- | --- |
| CHB-MIT | .52397/.92512 | .34542/.90605 | .38355/.91285 | **.41765 +/- .07678 / .91467 +/- .00789** |
| TUAB | .86708/.86289 | .87481/.87229 | .86892/.87379 | **.87027 +/- .00330 / .86966 +/- .00482** |
| TUEV | .63647/.80231 | .71484/.84858 | .71540/.84808 | **.68890 +/- .03708 / .83299 +/- .02169** |
| FACED | .37429/.45404 | .40240/.46739 | .34432/.42133 | **.37367 +/- .02372 / .44759 +/- .01935** |
| SEED-V | .22604/.38659 | .22359/.38433 | .21204/.37472 | **.22056 +/- .00610 / .38188 +/- .00515** |
| PhysioNet-MI | .51783/.63940 | .56074/.67021 | .54592/.66012 | **.54150 +/- .01779 / .65658 +/- .01283** |
| SHU-MI | .68458/.67286 | .67687/.65818 | .67267/.66568 | **.67804 +/- .00493 / .66557 +/- .00599** |
| BCIC2020-3 | .57167/.65717 | .56500/.65252 | .60500/.68432 | **.58056 +/- .01750 / .66467 +/- .01402** |
| Mumtaz2016 | .95662/.95016 | .96563/.96419 | .94831/.94269 | **.95685 +/- .00707 / .95235 +/- .00891** |
| MentalArithmetic | .68154/.89062 | .78280/.91855 | .71758/.86878 | **.72731 +/- .04191 / .89265 +/- .02037** |

### Finalized ISRUC five-seed result

This locked result uses seeds 42--46, bottom/right-only padding, and
`trunc_normal(std=.002)` classifier initialization. Every checkpoint was
selected by validation kappa and evaluated once on test. It replaces all
earlier ISRUC results in formal tables.

| Dataset | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean +/- population std |
| --- | --- | --- | --- | --- | --- | --- |
| ISRUC (kappa/F1) | .77291/.82287 | .77096/.81935 | .77465/.82294 | .76719/.81772 | .76654/.81563 | **.77045 +/- .00316 / .81970 +/- .00287** |

All five ISRUC kappa values exceed the published CBraMod reference
(.7442 +/- .0152), and the population standard deviation remains below .0032.

## Comparison with published CBraMod

CBraMod values are published references, not same-pipeline reruns. Primary
metrics are PR-AUC for binary tasks and kappa for multiclass tasks.

| Dataset | Published CBraMod | EfficientNet-B0 | Primary delta (pp) |
| --- | --- | --- | ---: |
| CHB-MIT | .3689/.8892 | **.41765 +/- .07678 / .91467 +/- .00789** | +4.88 |
| TUAB | **.9221/.9156** | .87027 +/- .00330 / .86966 +/- .00482 | -5.18 |
| TUEV | .6744/.8331 | **.68890 +/- .03708 / .83299 +/- .02169** | +1.45 |
| ISRUC | .7442/.8011 | **.77045 +/- .00316 / .81970 +/- .00287** | +2.63 |
| FACED | **.5041/.5618** | .37367 +/- .02372 / .44759 +/- .01935 | -13.04 |
| SEED-V | **.2569/.4101** | .22056 +/- .00610 / .38188 +/- .00515 | -3.63 |
| PhysioNet-MI | .5222/.6427 | **.54150 +/- .01779 / .65658 +/- .01283** | +1.93 |
| SHU-MI | **.7139/.6988** | .67804 +/- .00493 / .66557 +/- .00599 | -3.59 |
| BCIC2020-3 | .4216/.5383 | **.58056 +/- .01750 / .66467 +/- .01402** | +15.90 |
| Mumtaz2016 | **.9923/.9921** | .95685 +/- .00707 / .95235 +/- .00891 | -3.55 |
| MentalArithmetic | .6267/.7905 | **.72731 +/- .04191 / .89265 +/- .02037** | +10.06 |

The canonical B0 model exceeds the published CBraMod primary metric on six of
11 datasets and trails it on five. The largest gains are BCIC2020-3 (+15.90),
MentalArithmetic (+10.06), and CHB-MIT (+4.88) percentage points. FACED is the
largest deficit (-13.04). The published comparison is not a controlled paired
rerun and should be described as competitiveness evidence rather than a final
ranking.

## Same-pipeline EEGNet-8,2 control

EEGNet uses the same splits and preprocessing, validation-only selection, and
one final test evaluation per seed. Its control sweep uses seeds 3407--3409;
the ISRUC B0 margin below uses the locked five-seed B0 mean. EEGNet means are:

| Dataset | EEGNet-8,2 mean +/- std | B0 primary margin |
| --- | --- | ---: |
| CHB-MIT | .35091 +/- .08353 / .92546 +/- .00625 | +.06674 |
| TUAB | .86655 +/- .00441 / .87172 +/- .00132 | +.00372 |
| TUEV | .50123 +/- .05702 / .74625 +/- .02590 | +.18767 |
| ISRUC | .63655 +/- .03136 / .69791 +/- .02581 | +.13390 |
| FACED | .25136 +/- .01223 / .31909 +/- .01258 | +.12231 |
| SEED-V | .08174 +/- .00104 / .21567 +/- .00537 | +.13882 |
| PhysioNet-MI | .46778 +/- .00663 / .60074 +/- .00437 | +.07372 |
| SHU-MI | .67699 +/- .00333 / .68793 +/- .00215 | +.00105 |
| BCIC2020-3 | .07111 +/- .00208 / .24307 +/- .00320 | +.50945 |
| Mumtaz2016 | .94793 +/- .01276 / .93908 +/- .01043 | +.00892 |
| MentalArithmetic | .40375 +/- .01946 / .65731 +/- .02101 | +.32356 |

EfficientNet-B0 exceeds the same-pipeline EEGNet primary-metric mean on all 11
datasets, although the SHU-MI margin is only about .0011.

## Stability

The lowest primary-metric population standard deviations are ISRUC (.00316),
TUAB (.00330), and SHU-MI (.00493). CHB-MIT (.07678), MentalArithmetic
(.04191), and TUEV (.03708) remain seed-sensitive. In particular, the CHB-MIT
mean gain should not be described as uniformly reliable across seeds.

## Historical results and provenance

Earlier validation-selected phase-interleaved results remain useful as search
history but are superseded by the unified run whenever they differ in `P` or
AMP precision. Examples include CHB-MIT P=2 (.44264 PR-AUC), TUAB P=2
(.88461), SEED-V P=8 (.25653), PhysioNet-MI FP16 (.53334), SHU-MI FP16
(.68584), Mumtaz2016 P=2 (.96294), and MentalArithmetic P=2 (.71481). These
numbers must not be mixed into the canonical 11-dataset table.

Historical non-ISRUC launchers and logs remain under `experiments/` and
`experiments/logs/` for provenance. Superseded ISRUC log/checkpoint directories
were removed after locking the five-seed recipe; only
`isruc_p12_bottomrightpad_headstd002_5seed_v1` is retained. The matched
historical SHU-MI B5 ablation
(.69426 +/- .01483 PR-AUC) used the old FP16 B0 recipe and is not directly
comparable to the new all-BF16 headline value.
