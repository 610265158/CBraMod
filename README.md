# EEG-Vision

EEG-Vision studies whether standard ImageNet-pretrained convolutional networks
can serve as strong EEG encoders after a parameter-free, lossless change of
input geometry. Its phase-interleaved folding adapter maps
`[batch, channel, time]` into a single-channel 2D tensor while preserving every
sample exactly.

The repository is built from the CBraMod codebase and provides a unified,
validation-controlled comparison across 11 downstream EEG datasets. The main
backbone is EfficientNet-B0; EEGNet and selected timm backbones are available
as controlled baselines. The original CBraMod modules remain only where they
are required by legacy tasks.

Finalized metrics, protocol constraints, and CBraMod reference values are kept
in [`experiments/PHASE_FOLD_RESULTS.md`](experiments/PHASE_FOLD_RESULTS.md).

## Project Navigation

The codebase keeps the original CBraMod-compatible entrypoints, but the current
11-dataset downstream experiments should start from:

- `PROJECT_LAYOUT.md`: directory map and responsibilities.
- `experiments/README.md`: configured datasets, split counts, and tensor shapes.
- `experiments/downstream_11.py`: unified downstream runner.
- `experiments/run_downstream.sh`: unified shell entrypoint for downstream runs.
- `models/vision_model.py`: complete downstream vision model.
- `models/eeg_vision_adapter.py`: the single phase-folding adapter.
- `models/vision_backbone.py`: timm backbone, global pooling, stride, and checkpoint helpers.
- `datasets/shape_utils.py`: shared channel-by-time conversion and EEG
  clipping/scaling helpers.

Local training logs and checkpoints are written under `experiments/logs/` and
`experiments/checkpoints/`; these are ignored by git.

## Current experiment status

The temporal adapter now uses phase-interleaved folding:

```text
[B,C,T] -> reshape [B,1,C,T/P,P]
        -> permute [B,1,C,P,T/P]
        -> reshape [B,1,C*P,T/P]
```

For example, with `P=4`, the first folded row contains time indices
`0,4,8,12,...`. Results produced by the earlier contiguous-chunk adapter were
removed and must be rerun before reporting benchmark comparisons.

The finalized reporting seeds are `3407`, `3408`, and `3409`. Checkpoints are
selected using validation metrics, followed by one final test evaluation per
seed. Exploratory test-per-epoch peaks are not valid paper results.

## 🔨 Setup

Install Python and then install
[PyTorch](https://pytorch.org/get-started/locally/) separately for the CUDA
version on the target machine. PyTorch is intentionally not pinned in
`requirements.txt` because the correct wheel is platform dependent.

Install other requirements:

```commandline
pip install -r requirements.txt
``` 

## 🚢 Train

First verify that all configured external datasets are available:

```bash
bash experiments/run_downstream.sh --check_only --all
```

Run a CPU-only one-batch smoke test:

```bash
bash experiments/run_downstream.sh --dataset CHB-MIT --dry_run \
  --device cpu --batch_size 1 --num_workers 0 --random_init
```

Start one downstream run:

```bash
bash experiments/run_downstream.sh --dataset TUAB --cuda 0
```

Locked experiment launchers and complete YAML configs are documented under
`experiments/README.md`.
See [`experiments/README.md`](experiments/README.md) for exact shapes,
preprocessing, flags, and all 11 configured datasets.

The original CBraMod pretrained checkpoint is available on
[Hugging Face](https://huggingface.co/weighting666/CBraMod). It is used only by
the legacy CBraMod model path; the default vision path loads timm ImageNet
weights.

## References

1. Wang, J., Zhao, S., Luo, Z., Zhou, Y., Jiang, H., Li, S., ... & Pan, G. (2024). Cbramod: A criss-cross brain
   foundation model for eeg decoding. arXiv preprint arXiv:2412.07236.
2. Wang, X., Liu, X., Liu, X., Si, Q., Xu, Z., Li, Y., & Zhen, X. (2025, September). Eeg-dino: Learning eeg foundation
   models via hierarchical self-distillation. In International Conference on Medical Image Computing and
   Computer-Assisted Intervention (pp. 196-205). Cham: Springer Nature Switzerland.
3. Jiang, W. B., Zhao, L. M., & Lu, B. L. (2024). Large brain model for learning generic representations with tremendous
   EEG data in BCI. arXiv preprint arXiv:2405.18765.
