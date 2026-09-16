# Figure catalog

## figure-01-rerun-vs-published

- Purpose: compare each completed rerun group (val-selected test metric) with the published reference value.
- Plotted variables: grouped bars of the primary test metric; error bars are population std over seeds (rerun) and the published run-to-run std; annotations show rerun minus published.
- Key observation: to be read from the current numbers (see analysis-report.md).
- Interpretation checklist: check the sign of each delta, whether the rerun interval overlaps the published interval, and whether the gap is systematic across seeds.

## figure-02-val-curves

- Purpose: show validation dynamics per epoch (thin lines: individual seeds; thick line: mean).
- Plotted variables: validation selection metric (kappa for multiclass, PR-AUC for binary) versus epoch, one panel per dataset.
- Key observation: when each recipe plateaus and how much seed variance exists.
- Interpretation checklist: confirm runs stop at sensible early-stop points and that the mean curve is stable before the selected epoch.
