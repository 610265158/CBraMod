# Downstream Experiments

This folder keeps one source of truth for the prepared downstream datasets.

## Check

```bash
bash experiments/run_downstream.sh --list
bash experiments/run_downstream.sh --check_only --all
```

The current LMDB datasets require `lmdb` in the active Python environment:
FACED, SEED-V, PhysioNet-MI, SHU-MI, BCIC2020-3, Mumtaz2016,
and MentalArithmetic.

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

For locked runs, use the complete YAML directly. The YAML owns the dataset,
recipe, seed list, output roots, and result notes; no shell script duplicates
those values. The runner disables test-per-epoch evaluation for locked configs,
selects checkpoints on validation, and evaluates each selected checkpoint once
on the test split.

The retained locked convenience launchers are:

- `run_isruc_bottomrightpad_headstd002_5seed.sh`
- `run_tuev_p4_bs32_wd5e4_ep10_headstd002_5seed.sh`
- `run_physionet_mi_lr2e3_singlelr_ema995_5seed.sh`
- `run_bcic2020_3_bottomrightpad_5seed.sh`
- `run_mentalarithmetic_headstd002_ema995_5seed.sh`

## Complete experiment configs

For a reproducible run, prefer passing one complete YAML directly to the
runner. The YAML can contain `dataset`, `backbone_name`, `vision`, `training`,
`protocol`, and `output` sections:

```bash
bash experiments/run_downstream.sh \
  --config configs/backbones/efficientnet_b0/TUEV.yaml
```

Use the directory convention `configs/backbones/<backbone>/<dataset>.yaml` for
new locked recipes. A backbone-only `default.yaml` can hold shared defaults;
dataset files may extend it and record their five-seed results and notes.

CLI flags remain available as explicit temporary overrides. `--dataset` and
`--backbone_config` is retained for compatibility with older launchers.

Logs are saved under `./experiments/logs/vision/<dataset_name>/`.
Checkpoints are saved under `./experiments/checkpoints/<dataset_name>/`.

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
The finalized SHU-MI five-seed recipe overrides this default with clip
`[-1024,1024]` and `--shu_scale 32`.
Vision folding supports one or more temporal phases. For factor `P`, row
`c*P+p` contains `x[c, p::P]`; with `P=1` the time axis is unchanged. After
folding, inputs are
zero-padded on the bottom and right to multiples of the default CNN stride
(32), so CNN downsampling does not discard channel rows or trailing time
samples.
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

## Canonical Folded Shapes

| Dataset | P | Before padding | Backbone input |
| --- | ---: | --- | --- |
| CHB-MIT | 4 | `[B,1,64,500]` | `[B,1,64,512]` |
| TUAB | 4 | `[B,1,64,500]` | `[B,1,64,512]` |
| TUEV | 4 | `[B,1,64,250]` | `[B,1,64,256]` |
| ISRUC | 12 | `[B*20,1,72,500]` | `[B*20,1,96,512]` |
| FACED | 2 | `[B,1,64,1000]` | `[B,1,64,1024]` |
| SEED-V | 1 | `[B,1,62,200]` | `[B,1,64,224]` |
| PhysioNet-MI | 1 | `[B,1,64,800]` | `[B,1,64,800]` |
| SHU-MI | 2 | `[B,1,64,400]` | `[B,1,64,416]` |
| BCIC2020-3 | 1 | `[B,1,64,600]` | `[B,1,64,608]` |
| Mumtaz2016 | 4 | `[B,1,76,250]` | `[B,1,96,256]` |
| MentalArithmetic | 4 | `[B,1,80,250]` | `[B,1,96,256]` |

All canonical formal runs use BF16 AMP, validation-only checkpoint selection,
and one final test evaluation per seed. Exact values and provenance are in
`experiments/PHASE_FOLD_RESULTS.md`.
