#!/usr/bin/env python
import argparse
import csv
import importlib
import json
import math
import random
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from downstream_11 import EXPERIMENTS, safe_name, selected_names
except ModuleNotFoundError:
    from experiments.downstream_11 import EXPERIMENTS, safe_name, selected_names

DATASET_MODULES = {
    'CHB-MIT': 'datasets.chb_dataset',
    'TUAB': 'datasets.tuab_dataset',
    'TUEV': 'datasets.tuev_dataset',
    'ISRUC': 'datasets.isruc_dataset',
    'FACED': 'datasets.faced_dataset',
    'SEED-V': 'datasets.seedv_dataset',
    'PhysioNet-MI': 'datasets.physio_dataset',
    'SHU-MI': 'datasets.shu_dataset',
    'BCIC2020-3': 'datasets.speech_dataset',
    'Mumtaz2016': 'datasets.mumtaz_dataset',
    'MentalArithmetic': 'datasets.stress_dataset',
}


class RunningStats:
    def __init__(self, max_quantile_values, seed):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_value = math.inf
        self.max_value = -math.inf
        self.nan_count = 0
        self.inf_count = 0
        self.abs_gt_1 = 0
        self.abs_gt_10 = 0
        self.abs_gt_100 = 0
        self.abs_gt_1000 = 0
        self.max_quantile_values = max_quantile_values
        self.batch_quantile_values = max(10000, max_quantile_values // 8)
        self.quantile_values = np.empty(0, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def update(self, tensor):
        values = tensor.detach().float().cpu().reshape(-1)
        total = values.numel()
        if total == 0:
            return

        finite_mask = torch.isfinite(values)
        self.nan_count += torch.isnan(values).sum().item()
        self.inf_count += torch.isinf(values).sum().item()
        values = values[finite_mask]
        if values.numel() == 0:
            return

        self.count += values.numel()
        values64 = values.double()
        self.sum += values64.sum().item()
        self.sum_sq += values64.square().sum().item()
        self.min_value = min(self.min_value, values.min().item())
        self.max_value = max(self.max_value, values.max().item())

        abs_values = values.abs()
        self.abs_gt_1 += (abs_values > 1).sum().item()
        self.abs_gt_10 += (abs_values > 10).sum().item()
        self.abs_gt_100 += (abs_values > 100).sum().item()
        self.abs_gt_1000 += (abs_values > 1000).sum().item()
        self._sample_for_quantiles(values)

    def _sample_for_quantiles(self, values):
        if self.max_quantile_values <= 0:
            return
        values = values.numpy()
        take = min(len(values), self.batch_quantile_values)
        if take < len(values):
            values = values[self.rng.choice(len(values), size=take, replace=False)]
        self.quantile_values = np.concatenate([self.quantile_values, values.astype(np.float32, copy=False)])
        if len(self.quantile_values) > self.max_quantile_values:
            idx = self.rng.choice(len(self.quantile_values), size=self.max_quantile_values, replace=False)
            self.quantile_values = self.quantile_values[idx]

    def as_dict(self):
        if self.count == 0:
            return {
                'values': 0,
                'mean': None,
                'std': None,
                'min': None,
                'p01': None,
                'p05': None,
                'p25': None,
                'p50': None,
                'p75': None,
                'p95': None,
                'p99': None,
                'max': None,
                'abs_gt_1_pct': None,
                'abs_gt_10_pct': None,
                'abs_gt_100_pct': None,
                'abs_gt_1000_pct': None,
                'nan': self.nan_count,
                'inf': self.inf_count,
            }

        mean = self.sum / self.count
        variance = max(0.0, self.sum_sq / self.count - mean * mean)
        quantiles = {}
        if len(self.quantile_values):
            qs = np.percentile(
                self.quantile_values.astype(np.float64, copy=False),
                [1, 5, 25, 50, 75, 95, 99],
            )
            quantiles = {
                'p01': qs[0],
                'p05': qs[1],
                'p25': qs[2],
                'p50': qs[3],
                'p75': qs[4],
                'p95': qs[5],
                'p99': qs[6],
            }

        return {
            'values': self.count,
            'mean': mean,
            'std': math.sqrt(variance),
            'min': self.min_value,
            **quantiles,
            'max': self.max_value,
            'abs_gt_1_pct': self.abs_gt_1 / self.count * 100,
            'abs_gt_10_pct': self.abs_gt_10 / self.count * 100,
            'abs_gt_100_pct': self.abs_gt_100 / self.count * 100,
            'abs_gt_1000_pct': self.abs_gt_1000 / self.count * 100,
            'nan': self.nan_count,
            'inf': self.inf_count,
        }


def main():
    parser = argparse.ArgumentParser(description='Analyze dataloader value distributions.')
    parser.add_argument('--dataset', action='append', choices=sorted(EXPERIMENTS.keys()))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--splits', default='train,val,test',
                        help='comma-separated split names, for example train or train,val,test')
    parser.add_argument('--max_samples', type=int, default=512,
                        help='maximum records per split to scan')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_quantile_values', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--output_dir', default='experiments/reports')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    names = selected_names(args)
    splits = [split.strip() for split in args.splits.split(',') if split.strip()]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in names:
        print('Analyzing {}...'.format(name), flush=True)
        rows.extend(analyze_dataset(name, splits, args))

    csv_path = output_dir / 'data_distributions.csv'
    json_path = output_dir / 'data_distributions.json'
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('Wrote {}'.format(csv_path))
    print('Wrote {}'.format(json_path))


def analyze_dataset(name, splits, args):
    cfg = EXPERIMENTS[name]
    params = Namespace(
        downstream_dataset=name,
        datasets_dir=cfg['datasets_dir'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    module = importlib.import_module(DATASET_MODULES[name])
    loaders = module.LoadDataset(params).get_data_loader()
    rows = []
    for split in splits:
        if split not in loaders:
            continue
        rows.append(analyze_split(name, split, loaders[split], args))
    return rows


def analyze_split(name, split, loader, args):
    stats = RunningStats(args.max_quantile_values, args.seed)
    label_counts = Counter()
    shape = None
    samples_seen = 0
    batches_seen = 0

    for x, y in loader:
        if shape is None:
            shape = tuple(x.shape)
        stats.update(x)
        update_labels(label_counts, y)
        samples_seen += x.shape[0]
        batches_seen += 1
        if samples_seen >= args.max_samples:
            break

    row = {
        'dataset': name,
        'dataset_key': safe_name(name),
        'split': split,
        'first_batch_shape': str(shape),
        'samples_seen': samples_seen,
        'batches_seen': batches_seen,
        'label_counts': dict(sorted(label_counts.items())),
    }
    row.update(stats.as_dict())
    print(format_row(row), flush=True)
    return row


def update_labels(label_counts, y):
    values = y.detach().cpu().reshape(-1).tolist()
    for value in values:
        if isinstance(value, float):
            key = round(value, 6)
        else:
            key = int(value)
        label_counts[key] += 1


def write_csv(path, rows):
    fieldnames = [
        'dataset',
        'dataset_key',
        'split',
        'first_batch_shape',
        'samples_seen',
        'batches_seen',
        'values',
        'mean',
        'std',
        'min',
        'p01',
        'p05',
        'p25',
        'p50',
        'p75',
        'p95',
        'p99',
        'max',
        'abs_gt_1_pct',
        'abs_gt_10_pct',
        'abs_gt_100_pct',
        'abs_gt_1000_pct',
        'nan',
        'inf',
        'label_counts',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = row.copy()
            row['label_counts'] = json.dumps(row['label_counts'], sort_keys=True)
            writer.writerow(row)


def format_row(row):
    return (
        '{dataset:<16} {split:<5} shape={first_batch_shape:<18} '
        'n={samples_seen:<5} mean={mean:.5g} std={std:.5g} '
        'p01={p01:.5g} p50={p50:.5g} p99={p99:.5g} '
        'min={min:.5g} max={max:.5g} labels={label_counts}'
    ).format(**row)


if __name__ == '__main__':
    main()
