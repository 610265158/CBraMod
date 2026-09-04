# Project Layout

This repository contains the unified EEG-Vision downstream pipeline. Prepared
datasets are expected in a sibling `../BigDownstream/` directory by default;
the paths can be overridden from the command line.

## Main Entrypoints

| Path | Purpose |
| --- | --- |
| `experiments/downstream_11.py` | One runner for the 11 prepared downstream datasets. Use this for checks, dry runs, and training. |
| `finetune_main.py` | Low-level finetuning entrypoint used by the experiment runner. |
| `finetune_trainer.py` | Training loops for binary and multiclass tasks. |
| `finetune_evaluator.py` | Metric calculation for validation and test splits. |
| `experiments/run_downstream.sh` | Unified shell entrypoint for downstream experiments. |
| `configs/backbones/*/*.yaml` | Backbone profiles and dataset-specific locked configurations, including recipes, seeds, results, and notes where finalized. |
| `experiments/run_*5seed.sh` | Locked five-seed launchers for finalized recipes. |

Exploratory and per-dataset compatibility wrappers were removed from
`experiments/`; use `experiments/run_downstream.sh --config <yaml>` for new
runs.

## Dataset Code

| Path | Purpose |
| --- | --- |
| `datasets/*_dataset.py` | Dataset-specific dataloaders. Regular datasets return `[B, C, T]`. |
| `datasets/isruc_dataset.py` | ISRUC keeps the 20-epoch sequence dimension: `[B, 20, C, T]`, labels `[B, 20]`. |
| `datasets/shape_utils.py` | Shared helpers for converting processed patches into channel-by-time tensors. |
| `datasets/lmdb_utils.py` | Shared LMDB environment cache helpers. |

## Model Code

| Path | Purpose |
| --- | --- |
| `models/vision_model.py` | Complete downstream flow: input preparation, backbone, global pooling, and head. |
| `models/eeg_vision_adapter.py` | The single parameter-free phase-folding adapter; accepts both `[B,C,T]` and ISRUC `[B,S,C,T]`. |
| `models/vision_backbone.py` | timm creation and global pooling helpers. |

## Experiment Outputs

Local outputs are intentionally ignored by git:

| Path | Purpose |
| --- | --- |
| `experiments/logs/` | Local training logs from `experiments/downstream_11.py`. |
| `experiments/checkpoints/` | Local checkpoints from downstream experiments. |

## Current Notes

- `experiments/README.md` is the source of truth for the 11 configured
  downstream datasets, split counts, and dataloader shapes.
- Use `experiments/run_downstream.sh` or `experiments/downstream_11.py` for new
  downstream experiments instead of editing each wrapper separately.
- Supervised dataloaders share `datasets.shape_utils.clip_eeg`. The default is
  `float32`, clipped to `[-1024, 1024]`, then divided by `32`; SHU-MI currently
  overrides this with `clip[-512, 512]`, then divides by `64`.
- ISRUC is special: the loader keeps a batch of 20 consecutive sleep epochs and
  computes loss over all 20 labels. Do not flatten ISRUC into regular
  `[B, C, T]` records unless the model and loss are changed accordingly.
