#!/usr/bin/env python
"""Collect matched-pipeline foundation rerun results into CSV + appendix tables.

Parses the run logs under experiments/logs/foundation_rerun/, joins them with
their val-selected checkpoints, and reports per-run results plus per-(dataset,
model) summary statistics next to the published reference values.
"""
import argparse
import csv
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EPOCH_MULTI = re.compile(
    r'Epoch (\d+) : Training Loss: ([-\d.e+]+), ba: ([-\d.e+]+), kappa: ([-\d.e+]+), f1: ([-\d.e+]+)')
EPOCH_BINARY = re.compile(
    r'Epoch (\d+) : Training Loss: ([-\d.e+]+), ba: ([-\d.e+]+), pr_auc: ([-\d.e+]+), roc_auc: ([-\d.e+]+)')
TEST_MULTI = re.compile(r'Test Evaluation: ba: ([-\d.e+]+), kappa: ([-\d.e+]+), f1: ([-\d.e+]+)')
TEST_BINARY = re.compile(r'Test Evaluation: ba: ([-\d.e+]+), pr_auc: ([-\d.e+]+), roc_auc: ([-\d.e+]+)')
STOP = re.compile(r'Early stopping at epoch (\d+): (\w+) did not improve')
SAVE_EPOCH = re.compile(r'model save in .*epoch(\d+)_.*\.pth')
CMD = re.compile(r'--downstream_dataset (\S+).*?--lr (\S+).*?--batch_size (\d+)')

PUBLISHED = {
    ('FACED', 'cbramod'): {'BA': (0.5509, 0.0089), 'kappa': (0.5041, 0.0122), 'f1': (0.5618, 0.0093)},
    ('FACED', 'reve'): {'BA': (0.5646, 0.0164), 'kappa': (0.5080, 0.0191), 'f1': (0.5659, 0.0172)},
    ('TUAB', 'cbramod'): {'BA': None, 'PR_AUC': (0.9221, None), 'ROC_AUC': (0.9156, None)},
    ('TUAB', 'reve'): {'BA': (0.8315, 0.0014), 'PR_AUC': (0.9281, 0.0009), 'ROC_AUC': (0.9245, 0.0013)},
    ('HMC', 'cbramod'): {},
    ('HMC', 'reve'): {'BA': (0.7401, 0.0075), 'kappa': (0.6982, 0.0078), 'f1': (0.7638, 0.0074)},
    ('Mumtaz2016', 'cbramod'): {'BA': (0.9560, 0.0056), 'PR_AUC': (0.9923, 0.0032), 'ROC_AUC': (0.9921, 0.0025)},
    ('Mumtaz2016', 'reve'): {'BA': (0.9644, 0.0097), 'PR_AUC': (0.9961, 0.0013), 'ROC_AUC': (0.9957, 0.0015)},
    ('PhysioNet-MI', 'cbramod'): {'BA': (0.6417, None), 'kappa': (0.5222, 0.0169), 'f1': (0.6427, None)},
    ('PhysioNet-MI', 'reve'): {'BA': (0.6480, None), 'kappa': (0.5306, 0.0187), 'f1': (0.6484, None)},
}

SAFE_TO_DATASET = {
    'faced': 'FACED', 'tuab': 'TUAB', 'hmc': 'HMC', 'mumtaz2016': 'Mumtaz2016',
    'physionet_mi': 'PhysioNet-MI',
}


def read_text(path):
    return Path(path).read_text(encoding='utf-8', errors='replace').replace('\r', '')


def parse_run(log_path):
    text = read_text(log_path)
    rows = []
    for match in EPOCH_MULTI.finditer(text):
        epoch, loss, ba, kappa, f1 = match.groups()
        rows.append({'epoch': int(epoch), 'ba': float(ba), 'kappa': float(kappa), 'f1': float(f1)})
    binary = False
    if not rows:
        for match in EPOCH_BINARY.finditer(text):
            epoch, loss, ba, pr_auc, roc_auc = match.groups()
            rows.append({'epoch': int(epoch), 'ba': float(ba), 'pr_auc': float(pr_auc), 'roc_auc': float(roc_auc)})
        binary = True

    result = {'epochs_logged': len(rows), 'binary': binary}
    if rows:
        metric = 'pr_auc' if binary else 'kappa'
        best = max(rows, key=lambda row: row[metric])
        result['best_epoch'] = best['epoch']
        result['val_metric'] = metric
        result['val_best'] = best[metric]
        result['val_ba'] = best['ba']
    saves = SAVE_EPOCH.findall(text)
    if saves:
        # Final "model save in .../epochN_....pth" is the trainer's
        # last-improvement epoch; the printed-metric argmax can tie at 5dp.
        result['best_epoch'] = int(saves[-1])
    stop = STOP.search(text)
    if stop:
        result['stopped_epoch'] = int(stop.group(1))
    test = TEST_BINARY.search(text) if binary else TEST_MULTI.search(text)
    if test:
        values = [float(value) for value in test.groups()]
        result['test_ba'] = values[0]
        if binary:
            result['test_primary_name'] = 'pr_auc'
            result['test_primary'] = values[1]
            result['test_secondary_name'] = 'roc_auc'
            result['test_secondary'] = values[2]
        else:
            result['test_primary_name'] = 'kappa'
            result['test_primary'] = values[1]
            result['test_secondary_name'] = 'f1'
            result['test_secondary'] = values[2]
    command = CMD.search(text)
    if command:
        result['lr'] = command.group(2)
        result['batch_size'] = int(command.group(3))
    return result


def fmt(value, digits=4):
    if value in (None, ''):
        return '--'
    return '{:.{digits}f}'.format(value, digits=digits)


def summarize(records):
    groups = {}
    for record in records:
        if record['status'] != 'done':
            continue
        groups.setdefault((record['dataset'], record['model']), []).append(record)

    summaries = []
    for (dataset, model), rows in sorted(groups.items()):
        primary_name = rows[0]['primary_name']
        test_primary = [row['test_primary'] for row in rows]
        test_ba = [row['test_ba'] for row in rows]
        val_best = [row['val_best'] for row in rows]
        published = PUBLISHED.get((dataset, model), {})
        published_value = published_std = None
        entry = published.get('PR_AUC' if primary_name == 'pr_auc' else primary_name)
        if entry:
            published_value, published_std = entry
        mean_primary = statistics.mean(test_primary)
        summaries.append({
            'dataset': dataset,
            'model': model,
            'n': len(rows),
            'primary_name': primary_name,
            'val_best_mean': statistics.mean(val_best),
            'val_best_std': statistics.pstdev(val_best) if len(val_best) > 1 else 0.0,
            'test_ba_mean': statistics.mean(test_ba),
            'test_ba_std': statistics.pstdev(test_ba) if len(test_ba) > 1 else 0.0,
            'test_primary_mean': mean_primary,
            'test_primary_std': statistics.pstdev(test_primary) if len(test_primary) > 1 else 0.0,
            'published_value': published_value,
            'published_std': published_std,
            'diff': mean_primary - published_value if published_value is not None else None,
        })
    return summaries


def write_appendix(path, records, summaries, recipe='v1'):
    lines = []
    lines.append('# Foundation-model matched-pipeline rerun (CBraMod / REVE)')
    lines.append('')
    if recipe == 'v2':
        lines.append(
            'Protocol: the v2 unified recipe (learning rate 1e-4, warmup 3 epochs at factor 0.1, EMA 0.995, '
            'weight decay 5e-4, gradient clipping 1.0, early stop 10) with each dataset\'s epochs and '
            'selection metric from `configs/downstream.py`. Inputs are normalized by the dataset loaders '
            'with each model\'s released convention (CBraMod microvolt / 100; REVE per-dataset '
            '`scale_factor`; no clipping). Checkpoints are selected on the validation metric and evaluated '
            'once on test per seed (seeds 42-46).'
        )
    else:
        lines.append(
            'Protocol: per-dataset recipe from `configs/downstream.py` (unchanged); the only adapted '
            'hyperparameter is the learning rate (`--lr 1e-4`). Inputs are normalized by the dataset '
            'loaders with each model\'s released convention (CBraMod microvolt / 100; REVE per-dataset '
            '`scale_factor`; no clipping). Checkpoints are selected on the validation metric and '
            'evaluated once on test per seed.'
        )
    lines.append('')
    lines.append(
        'Models: CBraMod (4.92M parameters, full fine-tuning of the released checkpoint) and '
        'REVE-Base (69.19M parameters, full fine-tuning). No encoder is frozen.'
    )
    lines.append('')
    lines.append('### Deviations and provenance')
    lines.append('')
    if recipe == 'v2':
        lines.append('- Training recipe: the v2 unified recipe is applied to both models; the learning rate '
                     '(1e-4) remains the only per-model adaptation.')
    else:
        lines.append('- Training recipe: this repository\'s per-dataset configuration is used unchanged; the only adapted '
                     'hyperparameter is the learning rate (1e-4 for both models).')
    lines.append('- CBraMod: the released recipe uses a multi-LR setting (encoder 1e-4, head 1e-3*sqrt(batch/256)); '
                 'this rerun uses one uniform learning rate. AdamW in both cases.')
    lines.append('- REVE: the released procedure warm-starts with a linear probe before full fine-tuning, applies '
                 'mixup and StableAdamW, and the paper additionally describes LoRA and model souping; this rerun is a '
                 'single fine-tuning stage per seed with AdamW and no mixup/LoRA/souping.')
    lines.append('- Inputs: identical pre-processed benchmark arrays (inherited filters and splits); each loader '
                 'applies the model\'s released normalization (CBraMod microvolt/100; REVE per-dataset scale_factor; '
                 'no clipping). REVE\'s released TUAB pipeline uses a 21-channel memmap; this rerun feeds it the same '
                 '16-channel bipolar TUAB arrays as CBraMod.')
    lines.append('- Protocol: validation-selected checkpoint, one final test per seed, seeds 42-46, population std.')
    lines.append('')
    lines.append('## Suggested main-text sentence')
    lines.append('')
    dataset_phrase = 'five datasets' if recipe == 'v2' else 'three representative datasets'
    lines.append('> To assess the comparability of published foundation-model references, we additionally reran '
                 'CBraMod and REVE on {} under the inherited partitions and final-test '
                 'protocol. The rerun results and deviations from published values are reported in Appendix X.'.format(dataset_phrase))
    lines.append('')
    lines.append('## Summary (completed seeds, mean +/- population standard deviation)')
    lines.append('')
    lines.append('| Dataset | Model | Seeds | Rerun BA | Rerun primary | Published | Difference |')
    lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: |')
    for summary in summaries:
        primary_name = summary['primary_name']
        published = fmt(summary['published_value'])
        if summary['published_std'] is not None:
            published = '{} +/- {}'.format(published, fmt(summary['published_std']))
        lines.append('| {dataset} | {model} | {n} | {ba} +/- {ba_std} | {primary} +/- {primary_std} ({name}) | '
                     '{published} | {diff} |'.format(
                         dataset=summary['dataset'],
                         model=summary['model'],
                         n=summary['n'],
                         ba=fmt(summary['test_ba_mean']),
                         ba_std=fmt(summary['test_ba_std']),
                         primary=fmt(summary['test_primary_mean']),
                         primary_std=fmt(summary['test_primary_std']),
                         name=primary_name,
                         published=published,
                         diff=fmt(summary['diff']),
                     ))
    lines.append('')
    lines.append('## Per-run detail')
    lines.append('')
    lines.append('| Dataset | Model | Seed | Status | LR | Batch | Epochs | Best epoch | Val sel | Val best | '
                 'Test BA | Test primary | Test secondary | Published | Difference | Checkpoint |')
    lines.append('| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |')
    for record in records:
        lines.append('| {dataset} | {model} | {seed} | {status} | {lr} | {batch} | {epochs} | {best_epoch} | '
                     '{val_metric} | {val_best} | {test_ba} | {test_primary} | {test_secondary} | {published} | '
                     '{diff} | {checkpoint} |'.format(
                         dataset=record['dataset'],
                         model=record['model'],
                         seed=record['seed'],
                         status=record['status'],
                         lr=record['lr'],
                         batch=record['batch_size'],
                         epochs=record['epochs_logged'],
                         best_epoch=record['best_epoch'],
                         val_metric=record['val_metric'],
                         val_best=fmt(record['val_best']),
                         test_ba=fmt(record['test_ba']),
                         test_primary=fmt(record['test_primary']),
                         test_secondary=fmt(record['test_secondary']),
                         published=fmt(record['published_value']),
                         diff=fmt(record['diff']),
                         checkpoint=record['checkpoint'],
                     ))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root', default='experiments/logs/foundation_rerun')
    parser.add_argument('--checkpoint_root', default='experiments/checkpoints/foundation_rerun')
    parser.add_argument('--out', default='experiments/reports/foundation_rerun_results.csv')
    parser.add_argument('--appendix_out', default='experiments/reports/foundation_rerun_appendix.md')
    parser.add_argument('--recipe', choices=['v1', 'v2'], default='v1',
                        help='appendix provenance wording: v1 matched pipeline or the v2 unified recipe')
    args = parser.parse_args()

    log_root = ROOT / args.log_root
    checkpoint_root = ROOT / args.checkpoint_root
    latest = {}
    for log_path in log_root.glob('*/seed*/vision/*/*.log'):
        parts = log_path.parts
        model = parts[-5]
        seed = parts[-4]
        dataset = SAFE_TO_DATASET.get(parts[-2], parts[-2])
        key = (model, dataset, seed)
        if key not in latest or log_path.name > latest[key].name:
            latest[key] = log_path

    records = []
    for (model, dataset, seed), log_path in sorted(latest.items()):
        parsed = parse_run(log_path)
        safe_dataset = log_path.parts[-2]
        checkpoint = checkpoint_root / model / seed / safe_dataset / 'best.pth'
        published = PUBLISHED.get((dataset, model), {})
        primary_name = parsed.get('test_primary_name')
        published_value = None
        published_std = None
        diff = None
        if published and primary_name:
            entry = published.get('PR_AUC' if primary_name == 'pr_auc' else primary_name)
            if entry:
                published_value, published_std = entry
                diff = parsed['test_primary'] - published_value
        records.append({
            'dataset': dataset,
            'model': model,
            'seed': seed,
            'status': 'done' if 'test_primary' in parsed else 'partial',
            'lr': parsed.get('lr', ''),
            'batch_size': parsed.get('batch_size', ''),
            'epochs_logged': parsed.get('epochs_logged', 0),
            'best_epoch': parsed.get('best_epoch', ''),
            'val_metric': parsed.get('val_metric', ''),
            'val_best': parsed.get('val_best', ''),
            'primary_name': primary_name or '',
            'secondary_name': parsed.get('test_secondary_name', ''),
            'test_ba': parsed.get('test_ba', ''),
            'test_primary': parsed.get('test_primary', ''),
            'test_secondary': parsed.get('test_secondary', ''),
            'published_value': published_value if published_value is not None else '',
            'published_std': published_std if published_std is not None else '',
            'diff': diff if diff is not None else '',
            'checkpoint': str(checkpoint.relative_to(ROOT)) if checkpoint.exists() else 'missing',
            'log': str(log_path.relative_to(ROOT)),
        })

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['dataset', 'model', 'seed', 'status', 'lr', 'batch_size', 'epochs_logged', 'best_epoch',
                  'val_metric', 'val_best', 'primary_name', 'secondary_name', 'test_ba',
                  'test_primary', 'test_secondary', 'published_value', 'published_std', 'diff',
                  'checkpoint', 'log']
    with out_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summaries = summarize(records)
    appendix_path = ROOT / args.appendix_out
    write_appendix(appendix_path, records, summaries, recipe=args.recipe)

    print('wrote {} ({} runs) and {} ({} completed groups)'.format(
        out_path.relative_to(ROOT), len(records), appendix_path.relative_to(ROOT), len(summaries)))
    for record in records:
        print('{dataset:8s} {model:8s} seed {seed:6s} {status:8s} val_{val_metric} {val_best} test {test_primary} '
              'published {published_value} diff {diff}'.format(
                  dataset=record['dataset'], model=record['model'], seed=record['seed'],
                  status=record['status'], val_metric=record['val_metric'], val_best=record['val_best'],
                  test_primary=record['test_primary'], published_value=record['published_value'],
                  diff=record['diff']))
    for summary in summaries:
        print('SUMMARY {dataset:8s} {model:8s} n={n} test_{name} {mean} +/- {std} (published {published}, diff {diff})'.format(
            dataset=summary['dataset'], model=summary['model'], n=summary['n'],
            name=summary['primary_name'], mean=fmt(summary['test_primary_mean']),
            std=fmt(summary['test_primary_std']), published=fmt(summary['published_value']),
            diff=fmt(summary['diff'])), )


if __name__ == '__main__':
    main()
