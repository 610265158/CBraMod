# Downstream 11 Experiments

This folder keeps one source of truth for the 11 prepared downstream datasets.

## Check

```bash
bash experiments/run_downstream.sh --list
bash experiments/run_downstream.sh --check_only --all
```

The current LMDB datasets require `lmdb` in the active Python environment:
FACED, SEED-V, PhysioNet-MI, SHU-MI, BCIC2020-3, Mumtaz2016, and
MentalArithmetic.

## Dry Run

Use this on a login or CPU-only node to validate one batch and one forward pass:

```bash
bash experiments/run_downstream.sh --dataset CHB-MIT --dry_run --device cpu --batch_size 1 --num_workers 0 --random_init
```

## Train

Run one dataset:

```bash
bash experiments/run_downstream.sh --dataset CHB-MIT --cuda 0
```

The first phase-interleaved folding pilot compares `P=2` and `P=4` on three
small datasets with one seed:

```bash
bash experiments/run_phase_fold_small_pilot.sh
```

Common training parameters can be overridden on the runner command:

```bash
bash experiments/run_downstream.sh --dataset MentalArithmetic --cuda 0 \
  --lr 0.0005 --weight_decay 0.05 --batch_size 32 --epochs 50 \
  --label_smoothing 0.1 --dropout 0.1
```

Train-time left/right channel mirror augmentation is available for the 11
downstream datasets. It is disabled by default and only affects the training
loader:

```bash
bash experiments/run_downstream.sh --dataset CHB-MIT --cuda 0 \
  --mirror_augmentation true --mirror_prob 0.5
```

Circular time-roll augmentation is also training-only. Each regular sample is
shifted independently along time while all of its channels stay synchronized;
for ISRUC, each sleep epoch receives an independent shift. A max fraction of
`0.5` covers all unique circular offsets:

```bash
bash experiments/run_downstream.sh --dataset MentalArithmetic --cuda 0 \
  --time_roll_augmentation true --time_roll_prob 1.0 \
  --time_roll_max_fraction 0.5
```

Training defaults are resolved in this order: dataset defaults, then
model/backbone defaults, then command-line overrides. For example,
`--backbone_name convformer_s18` uses `lr=0.0001` by default because this
backbone is unstable at the EfficientNet default `lr=0.0005`; an explicit
`--lr` still takes precedence.

Arguments supported by `finetune_main.py` but not explicitly listed by the
runner can still be appended at the end; `experiments/downstream_11.py` forwards
unknown arguments to `finetune_main.py`.

The runner defaults to `HF_HUB_OFFLINE=1` for fast startup with cached timm
weights. Use `--online_weights` if a new machine needs to download the weights.

Run all 11 datasets sequentially:

```bash
bash experiments/run_downstream.sh --all --cuda 0
```

To reproduce the finalized EfficientNet-B0 recipes and reporting seeds, use
the consolidated runner below. With no dataset arguments it schedules all 11;
positional dataset names select a subset:

```bash

Recipes that were recently re-frozen and still require a clean three-seed
confirmation are marked `RERUN` by `--list`.  Their exact queue, parameters,
and per-dataset launcher commands are recorded in
`experiments/RERUN_QUEUE.md`.
bash experiments/run_finalized_efficientnet_b0_3seed.sh
bash experiments/run_finalized_efficientnet_b0_3seed.sh TUEV ISRUC
```

This runner disables test-per-epoch evaluation, selects checkpoints on the
validation primary metric, and evaluates each selected checkpoint once on the
test split. Historical search launchers remain in this directory for
provenance but are not the recommended reproduction entrypoint.

Logs are saved under `./experiments/logs/vision/<dataset_name>/`.
Checkpoints are saved under `./experiments/checkpoints/<dataset_name>/`.

Dataset-specific shell wrappers are kept in `experiments/scripts/` for
convenience, for example:

```bash
bash experiments/scripts/train_tuab.sh --cuda 0
```

## Configured Datasets

| Dataset | Task | Classes | Data path |
| --- | --- | ---: | --- |
| CHB-MIT | binary | 1 | `../BigDownstream/chb-mit/processed_seg` |
| TUAB | binary | 1 | `../BigDownstream/TUAB` |
| TUEV | multiclass | 6 | `../BigDownstream/TUEV_refine/processed` |
| ISRUC | multiclass | 5 | `../BigDownstream/ISRUC/precessed_filter_35` |
| FACED | multiclass | 9 | `../BigDownstream/faced/processed` |
| SEED-V | multiclass | 5 | `../BigDownstream/SEED-V/processed` |
| PhysioNet-MI | multiclass | 4 | `../BigDownstream/eeg-motor-movementimagery-dataset-1.0.0` |
| SHU-MI | binary | 1 | `../BigDownstream/shu_datasets` |
| BCIC2020-3 | multiclass | 5 | `../BigDownstream/speech/processed` |
| Mumtaz2016 | binary | 1 | `../BigDownstream/MDDPHCED/processed_lmdb_75hz` |
| MentalArithmetic | binary | 1 | `../BigDownstream/mental-arithmetic/processed` |

## Dataloader Shapes

All regular datasets now return channel-by-time tensors from the dataloader:
`[batch, channels, time]`. ISRUC keeps its sequence dimension because each
record has 20 epoch labels: `[batch, 20, channels, time]`.

Supervised dataset loaders use `datasets.shape_utils.clip_eeg`: values are
converted to `float32`, normally clipped to `[-1024, 1024]`, then divided by
`32`. SHU-MI is an explicit experiment-specific exception: it is clipped to
`[-512, 512]`, then divided by `--shu_scale` (default `64`).
An optional zero-phase Butterworth band-pass can be enabled for SHU-MI with
`--shu_bandpass_low` and `--shu_bandpass_high`. Filtering is applied to the
continuous `[32,800]` trial before clipping and scaling; for example,
`--shu_bandpass_low 1 --shu_bandpass_high 30 --shu_filter_order 4`.
Vision folding supports one or more temporal phases. For factor `P`, row
`c*P+p` contains `x[c, p::P]`; with `P=1` the time axis is unchanged. After
folding, inputs are
symmetrically zero-padded to the configured height stride and a width multiple
of 32, so stride-only CNN downsampling does not discard channel rows or trailing
time samples.
For dataset-level standardization experiments, pass the training-split-only
global scalar statistics, computed on raw clipped EEG before `/32`, with
`--eeg_dataset_mean` and `--eeg_dataset_std`.
The model then applies
`eeg_target_std * (x - mean) / std` without removing sample- or
channel-relative amplitude differences. `--eeg_target_std` defaults to `1`;
for example, pass `--eeg_target_std 32` to make the training-split global
input standard deviation approximately `32`.

The implementation uses one `PhaseFoldAdapter` for every dataset. It detects
regular `[B,C,T]` and ISRUC `[B,S,C,T]` inputs automatically; there is no
separate ISRUC folding algorithm. Override `P` with `--vision_fold_factor`.

| Dataset | Dataloader `x` shape |
| --- | --- |
| CHB-MIT | `[B, 16, 2000]` |
| TUAB | `[B, 16, 2000]` |
| TUEV | `[B, 16, 1000]` |
| ISRUC | `[B, 20, 6, 6000]` |
| FACED | `[B, 32, 2000]` |
| SEED-V | `[B, 62, 200]` |
| PhysioNet-MI | `[B, 64, 800]` |
| SHU-MI | `[B, 32, 800]` |
| BCIC2020-3 | `[B, 64, 600]` |
| Mumtaz2016 | `[B, 19, 1000]` |
| MentalArithmetic | `[B, 20, 1000]` |
