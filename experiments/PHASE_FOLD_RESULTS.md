# Phase-Interleaved Folding Results

This file records the formal phase-interleaved vision-transfer results. The
source of truth for the finalized EfficientNet-B0 numbers is the corresponding
YAML file under `configs/backbones/efficientnet_b0/`; this document mirrors
those values for quick inspection. Every formal checkpoint is selected using
validation data and evaluated once on test. Standard deviations are population
statistics, and the current formal recipes use seeds 42--46.

The adapter is the corrected phase-interleaved permutation

```text
I[c * P + p, w] = X[c, w * P + p]
```

Any checkpoint produced by the superseded contiguous-chunk adapter is stale and
must not be quoted.

## Formal 11-dataset EfficientNet-B0 results

The fold factor follows the geometry rule: choose the smallest valid `P` that
reaches at least 64 folded rows, while leaving SEED-V (`C=62`) at `P=1`.
All rows below use validation-selected checkpoints and one final test evaluation
per seed. Metrics are BA/PR-AUC/ROC-AUC for binary tasks and BA/kappa/F1 for
multiclass tasks.

| Dataset | Seeds | BA | PR-AUC / kappa | ROC-AUC / F1 |
| --- | --- | ---: | ---: | ---: |
| CHB-MIT | 42--46 | .75519 +/- .05985 | .40916 +/- .05359 | .89741 +/- .01944 |
| TUAB | 42--46 | .81839 +/- .00602 | .90627 +/- .00164 | .89840 +/- .00106 |
| TUEV | 42--46 | .64233 +/- .01636 | .69009 +/- .01896 | .83797 +/- .00760 |
| ISRUC | 42--46 | .80253 +/- .00272 | .77045 +/- .00316 | .81970 +/- .00287 |
| FACED | 42--46 | .55164 +/- .01790 | .49533 +/- .01970 | .55672 +/- .01626 |
| SEED-V | 42--46 | .36598 +/- .00622 | .20556 +/- .00779 | .36779 +/- .00570 |
| PhysioNet-MI | 42--46 | .64909 +/- .01251 | .53215 +/- .01672 | .64848 +/- .01328 |
| SHU-MI | 42--46 | .62734 +/- .00848 | .69819 +/- .01780 | .69622 +/- .01387 |
| BCIC2020-3 | 42--46 | .65387 +/- .01281 | .56733 +/- .01601 | .65386 +/- .01295 |
| Mumtaz2016 | 42--46 | .91527 +/- .00613 | .98637 +/- .00446 | .98459 +/- .00530 |
| MentalArithmetic | 42--46 | .69097 +/- .04184 | .79029 +/- .05364 | .87789 +/- .02513 |

Exact recipes, selected epochs, and per-seed values remain in the YAML files.
No prior 3407--3409 result is part of the formal table.

## Supplementary finalized experiments

HMC is an additional sleep-staging dataset outside the canonical 11-dataset
suite. Its EfficientNet-B0 five-seed result is BA/kappa/F1
`.75565 +/- .00189 / .70701 +/- .00231 / .77002 +/- .00173`. The HMC
ConvNeXt-Tiny DINOv3 GAP ablation is
`.75810 +/- .00552 / .71113 +/- .00627 / .77440 +/- .00433`.

The Mumtaz2016 amplitude-scale backbone comparison uses the simple no-EMA
recipe and is retained as a supplementary ablation:

| Backbone | BA | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: |
| EfficientNet-B0 | .91527 +/- .00613 | .98637 +/- .00446 | .98459 +/- .00530 |
| ConvNeXt-Tiny DINOv3 | .91760 +/- .01377 | .98417 +/- .00197 | .98245 +/- .00342 |
| ViT-Small DINOv3 | .90721 +/- .01520 | .98067 +/- .00161 | .97880 +/- .00213 |

The matched visual-pretraining controls on TUEV, PhysioNet-MI, ISRUC, and
SHU-MI are five-seed, validation-selected, single-final-test evaluations. The
multiclass controls (TUEV, PhysioNet-MI, ISRUC) report validation-selected
kappa; the binary SHU-MI control reports validation-selected PR-AUC:

| Backbone | TUEV pretrain kappa | TUEV random kappa | Physio pretrain kappa | Physio random kappa | ISRUC pretrain kappa | ISRUC random kappa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNet-B0 | .69009 +/- .01896 | .59945 +/- .03456 | .53215 +/- .01672 | .45734 +/- .01835 | .77045 +/- .00316 | .77328 +/- .00378 |
| ConvNeXt-Tiny DINOv3 | .70946 +/- .03046 | .50699 +/- .00936 | .54785 +/- .01053 | .26965 +/- .01190 | .79055 +/- .00176 | .63783 +/- .02164 |
| ViT-Small DINOv3 | .72625 +/- .00426 | .46146 +/- .01597 | .43634 +/- .00698 | .31522 +/- .00922 | .77854 +/- .00277 | .64486 +/- .02032 |

The binary SHU-MI control (validation PR-AUC selection):

| Backbone | SHU-MI pretrain PR-AUC | SHU-MI random PR-AUC |
| --- | ---: | ---: |
| EfficientNet-B0 | .69819 +/- .01780 | .51572 +/- .01750 |
| ConvNeXt-Tiny DINOv3 | .69096 +/- .01290 | .59832 +/- .00758 |
| ViT-Small DINOv3 | .66185 +/- .01610 | .63803 +/- .00423 |

Pretrained initialization improves every matched comparison except the
EfficientNet-B0 ISRUC pair, where random initialization already matches
ImageNet pretraining on the large ISRUC training set. The random-init recipes
live in `configs/ablation_random_init/`.

## Provenance policy

- Formal results come from YAML configs, not exploratory logs.
- `test_each_epoch=true` peaks and old three-seed sweeps are not formal results.
- Checkpoints are selected on validation PR-AUC for binary tasks or validation
  kappa for multiclass tasks, then evaluated once on test.
- Population standard deviation is used for every seed aggregate.
