# Project Layout

This repository keeps the original CBraMod components needed for compatibility
plus the unified EEG-Vision downstream and pretraining pipelines. Prepared
datasets are expected in a sibling `../BigDownstream/` directory by default;
the paths can be overridden from the command line.

## Main Entrypoints

| Path | Purpose |
| --- | --- |
| `experiments/downstream_11.py` | One runner for the 11 prepared downstream datasets. Use this for checks, dry runs, and training. |
| `finetune_main.py` | Low-level finetuning entrypoint used by the experiment runner. |
| `finetune_trainer.py` | Training loops for binary, multiclass, and regression tasks. |
| `finetune_evaluator.py` | Metric calculation for validation and test splits. |
| `pretrain_main.py` | EEG-Vision VICReg pretraining entrypoint. |
| `experiments/run_downstream.sh` | Unified shell entrypoint for downstream experiments. |
| `experiments/scripts/train_*.sh` | Backward-compatible dataset wrappers around `experiments/run_downstream.sh`. |

Two wrapper names come from the original project naming: `train_speech.sh`
runs `BCIC2020-3`, and `train_stress.sh` runs `MentalArithmetic`.

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
| `models/vision_backbone.py` | timm creation, global pooling, stride, and checkpoint helpers. |
| `models/eeg_vision_pretrain.py` | VICReg head and EEG augmentations on the shared encoder. |
| `models/legacy/model_for_*.py` | Legacy CBraMod wrappers for non-11-dataset tasks. |
| `models/cbramod.py` | Original CBraMod foundation model. |

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
- ISRUC is special: CBraMod keeps a batch of 20 consecutive sleep epochs and
  computes loss over all 20 labels. Do not flatten ISRUC into regular
  `[B, C, T]` records unless the model and loss are changed accordingly.
