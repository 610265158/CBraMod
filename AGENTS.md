# AGENTS.md

Repo-specific guidance for AI coding agents working in CBraMod (a.k.a. EEG-Vision).
Read this before running or editing anything. There are no tests, linter, or
typecheck config — verification = the `--dry_run` and `--check_only` steps below.

## Current experiment status (read first)

The EEG folding adapter was corrected on 25 Aug 2026 from contiguous-chunk to
**phase-interleaved** permutation (`x[c, p::P]` becomes row `c*P+p`). Every
`P>1` result/checkpoint produced before the fix is STALE and must be rerun
before quoting numbers. Only PhysioNet-MI (`P=1`) is unaffected.

- Source of truth for finalized numbers: `experiments/PHASE_FOLD_RESULTS.md`.
- Result policy (`paper/README.md`): formal tables may contain only
  validation-selected checkpoints with ONE final test evaluation per seed.
  Never copy exploratory `test_each_epoch=true` peaks into a table.
- Default reporting seeds are 3407, 3408, and 3409. Dataset-specific locked
  five-seed recipes in `configs/downstream.py:FINALIZED_FIVE_SEED_RECIPES`
  supersede those values; ISRUC uses seeds 42--46. All standard deviations are
  population statistics.

## Setup gotchas

- **`torch` is NOT in `requirements.txt`.** Install PyTorch separately per the
  wheel index for your CUDA version (e.g. `pip install torch --index-url
  https://download.pytorch.org/whl/cu121`), then `pip install -r requirements.txt`.
- `lmdb` is required and IS listed in `requirements.txt`; 7 of the 11 datasets
  are LMDB-backed (FACED, SEED-V, PhysioNet-MI, SHU-MI, BCIC2020-3, Mumtaz2016,
  MentalArithmetic). The other four use `pkl_split` (CHB-MIT, TUAB, TUEV) or
  `isruc_npy` (ISRUC). A `ModuleNotFoundError: lmdb` from
  `finetune_main.import_selected_module` means `pip install -r requirements.txt`
  was skipped in the active environment.
- Local run outputs under `experiments/logs/` and `experiments/checkpoints/`
  are gitignored. Checkpoint/log dirs are named via `safe_name()`: lowercase,
  `-`→`_` (so `--dataset CHB-MIT` writes to `experiments/checkpoints/chb_mit/`,
  not `CHB-MIT/`). Pre-existing subdirs like `experiments/checkpoints/small_lr1e3/`
  came from manual `--model_dir` overrides, not the default runner layout.

## Downstream experiments — the real entrypoint

The downstream path is two subprocess tiers. Do not call `finetune_main.py`
directly unless you are using a legacy dataset:

```
bash experiments/run_downstream.sh <flags>      # fixes cwd to repo root, then:
  -> python experiments/downstream_11.py <flags>  # builds the command, spawns:
       -> python finetune_main.py <translated flags + extra_args>
```

Critical implications:
- **Always invoke via `experiments/run_downstream.sh`**, never
  `python experiments/downstream_11.py` from another cwd — paths are resolved
  relative to repo root.
- `downstream_11.py` uses `parse_known_args` and forwards unknown args to
  `finetune_main.py`. **Any flag accepted by `finetune_main.py` works on the
  runner even if `run_downstream.sh --help` does not list it.** Check
  `finetune_main.py` for the full flag set (esp. `--classifier`,
  `--backbone_name`, `--vision_fold_factor`, `--multi_lr`, `--frozen`).
- The dataset name list passed to `--dataset` is `DOWNSTREAM_11_CONFIGS` keys
  in `configs/downstream.py` (11 names). The full registry in
  `finetune_main.py:DATASET_REGISTRY` adds 2 legacy datasets (`SEED-VIG`,
  `BCIC-IV-2a`) that the runner does NOT expose — those must call
  `finetune_main.py` directly with `--downstream_dataset` and use
  `models/legacy/model_for_*.py`.

### Shell wrapper name != dataset name

`experiments/scripts/train_*.sh` are thin `exec` wrappers around
`run_downstream.sh --dataset <NAME>`. Two names are misleading and inherited
from the original project:

| Wrapper | Actual `--dataset` value |
| --- | --- |
| `train_speech.sh` | `BCIC2020-3` |
| `train_stress.sh` | `MentalArithmetic` |

All others (`train_tuab.sh`, `train_tuev.sh`, `train_chb_mit.sh`, etc.) map
name-for-name. Always pass the uppercase dataset name when in doubt.

## Datasets live outside this repo — check first

All 11 datasets resolve to `../BigDownstream/<subdir>` (a sibling directory
OUTSIDE the repo). The runner will fail mid-flight if the path is missing.
Verify without training:

```bash
bash experiments/run_downstream.sh --check_only --all  # exits non-zero if missing
bash experiments/run_downstream.sh --list              # prints configured names + defaults
```

Configured dataset paths and per-dataset defaults live in
`configs/downstream.py:DOWNSTREAM_11_CONFIGS` — that is the source of truth,
not the README tables (which only summarize it).

## Smoke test before any change

CPU-only one-batch forward pass (no training, no CUDA, no network):

```bash
bash experiments/run_downstream.sh --dataset CHB-MIT --dry_run \
  --device cpu --batch_size 1 --num_workers 0 --random_init
```

`--random_init` sets `--use_pretrained_weights False` so the run works fully
offline. This is the closest thing this repo has to a unit test.

## Pretrained weights — three distinct sources

- `--use_pretrained_weights` (default True) → timm ImageNet weights for the
  `--backbone_name` backbone (`efficientnet_b0` by default). This is the
  default `--model_arch vision` path.
- `--vision_pretrained_checkpoint <pth>` → a VICReg backbone produced by
  `pretrain_main.py` (see "Pretraining vs downstream").
- `--foundation_dir` (default `pretrained_weights/pretrained_weights.pth`) →
  the original CBraMod foundation weights, from
  https://huggingface.co/weighting666/CBraMod. **Only the 2 legacy models
  (`models/legacy/model_for_*.py`) read this flag; the vision path ignores it.**

The runner sets `HF_HUB_OFFLINE=1` by default so timm loads cached weights
without network. On a fresh machine pass `--online_weights` once to allow the
HuggingFace download.

## Downstream model architecture

- `--model_arch vision` (default): the complete downstream model lives in
  `models/vision_model.py` and its dataset settings live in the `vision` block
  of `configs/downstream.py`. The single phase-interleaved folding adapter is
  in `models/eeg_vision_adapter.py`; it automatically accepts regular
  `[B,C,T]` and ISRUC `[B,S,C,T]` inputs. Timm-specific helpers are in
  `models/vision_backbone.py`.
- The fold factor `P` defaults to 2 (`DEFAULT_VISION`) and is overridden per
  dataset in `configs/downstream.py`. ISRUC's finalized recipe uses P=12.
  Override with `--vision_fold_factor`; the time length MUST be divisible by P
  or `PhaseFoldAdapter` raises.

Checkpoints write to `experiments/checkpoints/<safe_name>/`, and logs write to
`experiments/logs/vision/<safe_name>/` unless the runner roots are overridden
(`safe_name` lowercases and replaces `-`/space with `_`).

## Dataloader shape conventions (do not break)

- Regular datasets return `[B, C, T]` (channel-by-time).
- **ISRUC is the exception**: `[B, 20, C, T]` with labels `[B, 20]` because each
  record carries 20 consecutive sleep epochs and the loss is computed over all
  20. **Do not flatten ISRUC into `[B, C, T]` unless you also change the model
  and loss.**
- Every supervised loader funnels data through `datasets.shape_utils.clip_eeg`.
  The default is `np.float32` → clip to `[-1024, 1024]` → divide by `32`.
  **SHU-MI is the exception:** clip to `[-512, 512]` (`--shu_clip_limit`, default
  512) then divide by `--shu_scale` (default 64). Tuning found a divisor in
  `[32, 64]` clearly beats no divisor (AUROC ~+2pp) — SHU-MI's original
  no-divisor setting was the handicap. New datasets should use the default
  unless an explicitly documented experiment requires otherwise.
- Per-dataset exact dataloader shapes are in `experiments/README.md`.

## Loss / class-count rules (non-obvious)

Hardcoded in `finetune_trainer.py:Trainer.__init__` by dataset name:

| Task kind | Datasets | Criterion | `num_of_classes` in config |
| --- | --- | --- | ---: |
| binary | TUAB, CHB-MIT, SHU-MI, Mumtaz2016, MentalArithmetic | `BCEWithLogitsLoss` | **1** (single logit) |
| multiclass | FACED, SEED-V, PhysioNet-MI, ISRUC, BCIC2020-3, TUEV | `CrossEntropyLoss(label_smoothing=...)` | actual #classes |
| regression | SEED-VIG (legacy) | `MSELoss` | n/a |

Binary tasks use `classes=1` deliberately to match `BCEWithLogitsLoss`'s
single-logit contract (config sets `squeeze_binary=True`, so the model returns
`logits[..., 0]`). Not a bug. Adding a new binary dataset means updating the
`elif` chain in `finetune_trainer.py` or training will crash on criterion.

## Pretraining vs downstream

- `pretrain_main.py` + `pretrain_trainer.py` are THIS fork's VICReg
  self-supervised pretraining over the **vision backbone** (not the original
  CBraMod pipeline). `--dataset_dir` takes LMDB/pickle dirs (repeat the flag
  for TUSZ + TUAB); the produced backbone feeds downstream via
  `--vision_pretrained_checkpoint`. `models/eeg_vision_pretrain.py` holds the
  VICReg head and augmentations.
- The original CBraMod foundation model still lives in `models/cbramod.py` +
  `models/criss_cross_transformer.py` but is only wired into the 2 legacy
  dataset wrappers (`models/legacy/model_for_seedvig.py`,
  `models/legacy/model_for_bciciv2a.py`) via `--foundation_dir`.
- Default seed is `3407` on both paths. **`finetune_main.py --seed` help text
  says "(default: 0)" but the actual default is 3407 — trust the code, not the
  help string.** Preserve 3407 for reproducibility comparisons.

## TUAB / TUEV split reproducibility

`preprocessing/README.md` documents that older TUAB/TUEV splits inherited from
BIOT/LaBraM were non-deterministic across hardware. The current preprocessing
scripts produce fixed splits. Use the repo's preprocessing scripts, not the
upstream BIOT/LaBraM versions, if you need comparable numbers. Sample counts
for TUAB v3.0.1 / TUEV v2.0.0 / TUEV v2.0.1 are listed there and will differ
from upstream — do not quote our sample counts for other versions.

## Editing conventions

- New downstream dataset → add an entry in
  `configs/downstream.py:DOWNSTREAM_11_CONFIGS` (use the `_dataset(...)` helper)
  and a matching `datasets/<name>_dataset.py` exposing `LoadDataset(params)` with
  `get_data_loader()`. The runner picks them up automatically.
- New training hyperparameter → add to `TRAINING_KEYS` in
  `configs/downstream.py`, to `DEFAULT_TRAINING`, to `finetune_main.py`'s
  argparse, and (if surfaced) to `experiments/downstream_11.py`'s argparse. All
  four must agree or `downstream_11.py:append_optional_training_args` will skip it.
- Keep the `[B, C, T]` contract in any new dataloader.
