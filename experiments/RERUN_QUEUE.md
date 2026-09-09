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
