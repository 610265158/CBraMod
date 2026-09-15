# Reproduction Queue

The formal 11-dataset geometry reproduction is complete. All current formal
EfficientNet-B0 results use the five-seed recipes recorded in
`configs/backbones/efficientnet_b0/*.yaml`; there are no pending three-seed
rows in the paper table.

Use the complete YAML-driven entrypoint for any rerun:

```bash
bash experiments/run_downstream.sh --config configs/backbones/efficientnet_b0/<dataset>.yaml
```

Each formal recipe uses validation-only checkpoint selection and one final test
evaluation per seed. Exploratory `test_each_epoch=true` runs and superseded
3407--3409 sweeps are historical artifacts and must not replace the YAML
results.

## Completed: ViT-Small fold-factor sweep under the corrected recipe

The corrected-recipe rerun of `configs/ablation_p/vit_small_dinov3/` (CHB-MIT,
TUEV, MentalArithmetic; `P in {1,2,8}`, with `P=4` reused from the finalized
main recipes) completed on 2026-09-15: 45/45 runs, zero failures
(`experiments/run_vit_p_sweep_clstoken.sh`). The nine sweep configs now record
full per-seed results. Paper-appendix fold-factor tables still carry the
superseded flatten@1e-4 numbers and await a separate update.
