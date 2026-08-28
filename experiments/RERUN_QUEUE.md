# Fixed-Recipe Rerun Queue

These datasets had their default training recipes re-frozen during the
EMA/bf16 stability pass.  Keep them marked `RERUN` until a clean reproduction
over seeds 3407, 3408, and 3409 has been reviewed.  Each checkpoint must be
selected on validation and evaluated on test exactly once.

Run the queue one dataset at a time with:

```bash
CUDA_ID=0 bash experiments/run_fixed_recipe_reruns.sh BCIC2020-3
CUDA_ID=0 bash experiments/run_fixed_recipe_reruns.sh Mumtaz2016
CUDA_ID=0 bash experiments/run_fixed_recipe_reruns.sh MentalArithmetic
```

## Queue

- [ ] **BCIC2020-3 — `fixed_recipe_v2`**
  - Input/fold: `[B,64,600]`, `P=2` -> `[B,1,128,300]`
  - Train: 30 epochs, batch 32, lr `1e-3`, wd `5e-3`, warmup 3
  - Stability: EMA `.995`, bf16, grad clip 1
  - Augmentation: time roll `p=1`, max fraction `.5`; no mirror
  - Selection: validation kappa

- [ ] **Mumtaz2016 — `fixed_recipe_v2`**
  - Input/fold: `[B,19,1000]`, `P=2` -> `[B,1,38,500]`
  - Train: 30 epochs, batch 32, lr `1e-3`, wd `5e-4`, warmup 3
  - Stability: EMA `.995`, bf16
  - Augmentation: time roll `p=1`, max fraction `.5`; no mirror
  - Selection: validation PR-AUC

- [ ] **MentalArithmetic — `fixed_recipe_v2`**
  - Input/fold: `[B,20,1000]`, `P=2` -> `[B,1,40,500]`
  - Train: 30 epochs, batch 64, lr `1e-3`, wd `5e-4`, warmup 3
  - Stability: EMA `.99`, bf16, weighted BCE `pos_weight=3`
  - Augmentation: time roll `p=.5`, max fraction `.25`; mirror `p=.5`
  - Selection: validation PR-AUC
  - Existing reference reproduction: PR-AUC `.71481 +/- .03636`, AUROC
    `.83700 +/- .00403`; rerun remains queued for a clean end-to-end check.

`bash experiments/run_downstream.sh --list` displays `RERUN` beside these
three datasets.  After reviewing all three seeds, update
`experiments/PHASE_FOLD_RESULTS.md`, then remove the dataset from
`RERUN_REQUIRED_DATASETS` in `configs/downstream.py`.
