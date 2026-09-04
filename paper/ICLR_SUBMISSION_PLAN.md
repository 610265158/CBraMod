# ICLR submission plan

## Core claim

The strongest defensible claim is:

> A parameter-free, invertible temporal folding operation aligns raw EEG with
> hierarchical 2D CNN locality, making a compact ImageNet encoder competitive
> with an EEG foundation model under full fine-tuning.

Avoid the broader claim that CNNs universally outperform EEG foundation
models. The current evidence does not support it, and frozen/few-shot transfer
is likely to favor large EEG-pretrained encoders.

## What makes this an ICLR paper rather than only a benchmark report

1. A formal geometry view: folding remaps temporal offsets, changes signal
   occupancy after padding, and delays height collapse.
2. Falsifiable predictions: gains should depend on channel count, fold factor,
   sampling rate, and the importance of physical electrode topology.
3. Controls that test the induced neighborhood: phase-order permutation and
   channel-order permutation.
4. Broad evidence across tasks rather than one application.
5. Clear negative results, especially FACED, that expose the boundary of the
   proposed inductive bias.

## Priority 0: required for submission

- Reproduce CBraMod under the already-matched split definitions, with the same
  preprocessing, seeds, epoch budget, and validation-only selection.
- Run EfficientNet-B0 with random initialization.
- Run a frozen ImageNet linear probe.
- Freeze a new protocol without consulting test results and repeat at least a
  representative set of datasets.
- Report parameter count, FLOPs, throughput, memory, and wall time.
- Verify every dataset citation, license, version, subject split, and sample
  count.

## Priority 1: experiments that strengthen novelty

- Full `P = 1, 2, 4, 8` ablation on TUEV, FACED, PhysioNet-MI,
  BCIC2020-3, and MentalArithmetic.
- Phase-order permutation within each channel.
- Channel-order permutation and physical-coordinate augmentation.
- Add matched supervised 1D CNN controls only if space permits; keep the main
  narrative focused on foundation models versus generic vision transfer.
- Subject-level bootstrap confidence intervals.

## Priority 2: high-upside extension

- Add electrode coordinates or channel-ID planes while retaining lossless
  folding.
- Pretrain the visual encoder on TUSZ and TUAB with masked raw-wave
  reconstruction, excluding every downstream evaluation subject.
- Evaluate frozen transfer, few-shot learning, missing channels, and unseen
  montages.
- Test whether a rule derived from `(C, T, sampling rate)` can
  choose the fold factor on a held-out dataset.

## Likely reviewer challenges

- "This is only a reshape." Answer with the bijection, receptive-field
  remapping, padding occupancy analysis, and permutation controls.
- "The comparison is unfair." Answer with same-pipeline CBraMod three-seed
  runs and split definitions matching the published CBraMod protocol.
- "Hyperparameters were tuned on test." Answer only after a prospective rerun;
  prose cannot repair leakage.
- "ImageNet semantics are irrelevant to EEG." Separate architecture from
  initialization using random-init and linear-probe controls.
- "The method ignores electrode topology." Present this as a measured
  limitation and add coordinate/channel-order experiments.
- "Wins on only part of the benchmark." Emphasize the scientific question:
  when does generic visual locality suffice, and when is domain structure
  necessary?

## Suggested paper narrative

1. Foundation-model gains confound pretraining, architecture, and input
   geometry.
2. Direct EEG images are geometrically mismatched to hierarchical CNNs.
3. A lossless fold changes locality without changing information.
4. A compact ImageNet CNN is competitive across a broad suite.
5. Geometry and permutation analyses characterize the induced neighborhood.
6. Negative tasks identify the need for physical topology and EEG pretraining.
