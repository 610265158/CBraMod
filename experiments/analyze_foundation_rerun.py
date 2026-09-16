#!/usr/bin/env python
"""Strict analysis bundle for the foundation-model matched-pipeline rerun.

Reads the results CSV + run logs under experiments/ and produces:
  experiments/reports/analysis/analysis-report.md
  experiments/reports/analysis/stats-appendix.md
  experiments/reports/analysis/figure-catalog.md
  experiments/reports/analysis/figures/figure-01-rerun-vs-published.{png,pdf}
  experiments/reports/analysis/figures/figure-02-val-curves.{png,pdf}
"""
import csv
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_foundation_results as collector  # noqa: E402

OUT = ROOT / 'experiments/reports/analysis'
FIG = OUT / 'figures'

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def load_records():
    csv_path = ROOT / 'experiments/reports/foundation_rerun_results.csv'
    if not csv_path.exists():
        raise SystemExit('Missing {}. Run experiments/collect_foundation_results.py first.'.format(csv_path))
    with csv_path.open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def val_series(log_path):
    text = collector.read_text(log_path)
    values = []
    for match in collector.EPOCH_MULTI.finditer(text):
        values.append(float(match.group(4)))
    if values:
        return values
    return [float(match.group(4)) for match in collector.EPOCH_BINARY.finditer(text)]


def t_interval(samples):
    n = len(samples)
    if n < 2:
        return None
    mean = statistics.mean(samples)
    sd = statistics.stdev(samples)
    t = T95.get(n - 1, 1.96)
    half = t * sd / math.sqrt(n)
    return mean - half, mean + half


def published_entry(dataset, model, metric_name):
    published = collector.PUBLISHED.get((dataset, model), {})
    return published.get('PR_AUC' if metric_name == 'pr_auc' else metric_name)


def group_done(records):
    groups = {}
    for record in records:
        if record['status'] != 'done':
            continue
        groups.setdefault((record['dataset'], record['model']), []).append(record)
    return groups


def figure_main(groups):
    labels, rerun_means, rerun_stds, pub_means, pub_stds = [], [], [], [], []
    for (dataset, model), rows in sorted(groups.items()):
        metric = rows[0]['primary_name']
        values = [float(row['test_primary']) for row in rows]
        entry = published_entry(dataset, model, metric)
        if not entry:
            continue
        labels.append('{}\n{} ({})'.format(dataset, model, 'kappa' if metric == 'kappa' else 'PR-AUC'))
        rerun_means.append(statistics.mean(values))
        rerun_stds.append(statistics.pstdev(values) if len(values) > 1 else 0.0)
        pub_means.append(entry[0])
        pub_stds.append(entry[1] or 0.0)

    fig, ax = plt.subplots(figsize=(max(5.5, 2.4 * len(labels)), 4.2))
    xs = list(range(len(labels)))
    width = 0.36
    ax.bar([x - width / 2 for x in xs], rerun_means, width, yerr=rerun_stds, capsize=4,
           label='Rerun (this repo, val-selected, per-seed)')
    ax.bar([x + width / 2 for x in xs], pub_means, width, yerr=pub_stds, capsize=4,
           label='Published reference')
    for x, rerun, pub in zip(xs, rerun_means, pub_means):
        ax.text(x, max(rerun, pub) + 0.008, '{:+.3f}'.format(rerun - pub), ha='center', fontsize=8)
    ax.set_ylabel('primary test metric')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'figure-01-rerun-vs-published.png', dpi=200)
    fig.savefig(FIG / 'figure-01-rerun-vs-published.pdf')
    plt.close(fig)


def figure_curves(groups):
    datasets = sorted({dataset for dataset, _ in groups})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.6), squeeze=False)
    colors = {'reve': 'tab:blue', 'cbramod': 'tab:orange'}
    for ax, dataset in zip(axes[0], datasets):
        for model in sorted({model for d, model in groups if d == dataset}):
            rows = groups[(dataset, model)]
            longest = 0
            series_list = []
            for row in rows:
                series = val_series(ROOT / row['log'])
                series_list.append(series)
                longest = max(longest, len(series))
            for series in series_list:
                ax.plot(range(1, len(series) + 1), series, color=colors.get(model, 'gray'), alpha=0.35, lw=1)
            mean_curve = []
            for epoch in range(longest):
                epoch_values = [series[epoch] for series in series_list if len(series) > epoch]
                mean_curve.append(statistics.mean(epoch_values))
            ax.plot(range(1, longest + 1), mean_curve, color=colors.get(model, 'gray'), lw=2,
                    label='{} mean (n={})'.format(model, len(rows)))
        ax.set_title(dataset)
        ax.set_xlabel('epoch')
        ax.set_ylabel('validation selection metric')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'figure-02-val-curves.png', dpi=200)
    fig.savefig(FIG / 'figure-02-val-curves.pdf')
    plt.close(fig)


def write_reports(records, groups):
    lines = []
    lines.append('# Rerun analysis: CBraMod / REVE matched-pipeline rerun')
    lines.append('')
    lines.append('Analysis question: under this repository\'s own training recipe (learning rate adapted to 1e-4), '
                 'how do the reruns of CBraMod and REVE compare with the published reference values on the shared '
                 'benchmark partitions?')
    lines.append('')
    lines.append('## Key findings')
    lines.append('')
    summary_rows = []
    for (dataset, model), rows in sorted(groups.items()):
        metric = rows[0]['primary_name']
        primary = [float(row['test_primary']) for row in rows]
        ba = [float(row['test_ba']) for row in rows]
        entry = published_entry(dataset, model, metric)
        ci = t_interval(primary)
        delta = statistics.mean(primary) - entry[0] if entry else None
        summary_rows.append((dataset, model, len(rows), metric, primary, ba, entry, ci, delta))
        if entry:
            published_text = '{:.4f} +/- {}'.format(
                entry[0], '{:.4f}'.format(entry[1]) if entry[1] is not None else 'n/a')
            delta_text = ' (95% CI [{:.4f}, {:.4f}])'.format(ci[0], ci[1]) if ci else ''
            lines.append('- **{} / {}** (n={}): {} = {:.4f} +/- {:.4f} (population std) vs published {} '
                         '-> delta {:+.4f}{}.'.format(
                             dataset, model, len(rows), metric,
                             statistics.mean(primary), statistics.pstdev(primary) if len(primary) > 1 else 0.0,
                             published_text, delta, delta_text))
        else:
            lines.append('- **{} / {}** (n={}): {} = {:.4f} +/- {:.4f} (population std); no published reference '
                         'for this dataset.'.format(
                             dataset, model, len(rows), metric,
                             statistics.mean(primary), statistics.pstdev(primary) if len(primary) > 1 else 0.0))
    lines.append('')
    lines.append('## Caveats and blockers')
    lines.append('')
    lines.append('- The published reference is a mean +/- run-to-run std from the original authors (their aggregation '
                 'details, e.g. model souping, are not fully known); the rerun is one val-selected run per seed under '
                 'this repository\'s protocol, so no paired significance test is valid. We report intervals and '
                 'standardized gaps instead.')
    lines.append('- Recipe deviations are documented in experiments/reports/foundation_rerun_appendix.md '
                 '(no linear-probe warm start, no mixup/StableAdamW/LoRA/souping for REVE; uniform LR for CBraMod).')
    lines.append('- Dataset coverage is incomplete while the grid runs; groups with n<2 seeds have no interval.')
    lines.append('')
    (OUT / 'analysis-report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    stats = []
    stats.append('# Stats appendix')
    stats.append('')
    stats.append('| Dataset | Model | n | Metric | Rerun mean | Rerun std (pop) | 95% CI (t) | Published | Published std '
                 '| Delta | Delta / published std |')
    stats.append('| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |')
    for dataset, model, n, metric, primary, ba, entry, ci, delta in summary_rows:
        mean = statistics.mean(primary)
        std = statistics.pstdev(primary) if len(primary) > 1 else 0.0
        ci_text = '[{:.4f}, {:.4f}]'.format(ci[0], ci[1]) if ci else '--'
        ratio = delta / entry[1] if entry and entry[1] else float('nan')
        stats.append('| {} | {} | {} | {} | {:.4f} | {:.4f} | {} | {:.4f} | {} | {:+.4f} | {:+.1f} |'.format(
            dataset, model, n, metric, mean, std, ci_text,
            entry[0] if entry else float('nan'), '{:.4f}'.format(entry[1]) if entry and entry[1] else '--',
            delta if delta is not None else float('nan'), ratio))
    stats.append('')
    stats.append('Test-time BA per group: ' + '; '.join(
        '{} / {} = {:.4f} +/- {:.4f}'.format(dataset, model, statistics.mean(ba),
                                             statistics.pstdev(ba) if len(ba) > 1 else 0.0)
        for dataset, model, _, _, _, ba, _, _, _ in summary_rows))
    stats.append('')
    stats.append('Blockers: published std is run-to-run variability, not the standard error of the comparison; '
                 'no seed-level pairing exists against the published aggregate; groups with n=1 have no interval.')
    (OUT / 'stats-appendix.md').write_text('\n'.join(stats) + '\n', encoding='utf-8')

    catalog = []
    catalog.append('# Figure catalog')
    catalog.append('')
    catalog.append('## figure-01-rerun-vs-published')
    catalog.append('')
    catalog.append('- Purpose: compare each completed rerun group (val-selected test metric) with the published '
                   'reference value.')
    catalog.append('- Plotted variables: grouped bars of the primary test metric; error bars are population std over '
                   'seeds (rerun) and the published run-to-run std; annotations show rerun minus published.')
    catalog.append('- Key observation: to be read from the current numbers (see analysis-report.md).')
    catalog.append('- Interpretation checklist: check the sign of each delta, whether the rerun interval overlaps the '
                   'published interval, and whether the gap is systematic across seeds.')
    catalog.append('')
    catalog.append('## figure-02-val-curves')
    catalog.append('')
    catalog.append('- Purpose: show validation dynamics per epoch (thin lines: individual seeds; thick line: mean).')
    catalog.append('- Plotted variables: validation selection metric (kappa for multiclass, PR-AUC for binary) versus '
                   'epoch, one panel per dataset.')
    catalog.append('- Key observation: when each recipe plateaus and how much seed variance exists.')
    catalog.append('- Interpretation checklist: confirm runs stop at sensible early-stop points and that the mean '
                   'curve is stable before the selected epoch.')
    (OUT / 'figure-catalog.md').write_text('\n'.join(catalog) + '\n', encoding='utf-8')


def main():
    (FIG).mkdir(parents=True, exist_ok=True)
    records = load_records()
    groups = group_done(records)
    if not groups:
        raise SystemExit('No completed runs yet; nothing to analyze.')
    figure_main(groups)
    figure_curves(groups)
    write_reports(records, groups)
    print('analysis bundle written to {} ({} groups, {} completed runs)'.format(
        OUT.relative_to(ROOT), len(groups), sum(len(rows) for rows in groups.values())))


if __name__ == '__main__':
    main()
