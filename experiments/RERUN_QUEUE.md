# Reproduction Queue

The unified min-64 geometry reproduction is complete. There are currently no
required geometry reruns in the formal 11-dataset table.

## Completed: `all11_min64_repro_3seed_v1`

- [x] CHB-MIT, `P=4`, seeds 3407/3408/3409
- [x] TUAB, `P=4`, seeds 3407/3408/3409
- [x] TUEV, `P=4`, seeds 3407/3408/3409
- [x] ISRUC, `P=12`, seeds 42/43/44/45/46, bottom/right padding, head std .002
- [x] FACED, `P=2`, seeds 3407/3408/3409
- [x] SEED-V, `P=1`, seeds 3407/3408/3409
- [x] PhysioNet-MI, `P=1`, seeds 3407/3408/3409
- [x] SHU-MI, `P=2`, seeds 3407/3408/3409
- [x] BCIC2020-3, `P=1`, seeds 3407/3408/3409
- [x] Mumtaz2016, `P=4`, seeds 3407/3408/3409
- [x] MentalArithmetic, `P=4`, seeds 3407/3408/3409

All runs use EfficientNet-B0 ImageNet initialization and BF16 AMP. ISRUC uses
its locked five-seed recipe; the remaining rows in this historical unified
sweep use seeds 3407/3408/3409. Each best
checkpoint was selected only on validation and tested exactly once. Exact
recipes and results are recorded in `experiments/PHASE_FOLD_RESULTS.md`.

The historical hard-coded launcher was removed; use the complete YAML configs
and the unified `experiments/run_downstream.sh` entrypoint for reruns.

Logs: `experiments/logs/all11_min64_repro_3seed_v1/`

Historical fixed-recipe, P-search, balancing, and test-each-epoch probes were
removed from `experiments/`; their summarized provenance remains in
`experiments/PHASE_FOLD_RESULTS.md` and the paper notes.
