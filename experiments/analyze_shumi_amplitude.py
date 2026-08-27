#!/usr/bin/env python
"""Exhaustive SHU-MI amplitude statistics with sampled quantiles and plots."""

import argparse
import csv
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import lmdb
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.channel_mirror import STANDARD_32_CHANNELS


CLIP_LIMIT = 512.0
MODEL_DIVISOR = 1.0
SAMPLED_VALUES_PER_TRIAL = 256


class ExactMoments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values):
        values = np.asarray(values)
        self.count += values.size
        self.total += float(np.sum(values, dtype=np.float64))
        self.total_sq += float(np.sum(np.square(values, dtype=np.float64), dtype=np.float64))
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))

    def result(self):
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            'count': self.count,
            'mean': mean,
            'std': math.sqrt(variance),
            'min': self.minimum,
            'max': self.maximum,
        }


def subject_from_key(key):
    match = re.search(r'sub-(\d+)', key)
    return int(match.group(1)) if match else -1


def percentiles(values, quantiles):
    values = np.asarray(values, dtype=np.float64)
    result = np.percentile(values, quantiles)
    return {f'p{str(q).replace(".", "p")}': float(v) for q, v in zip(quantiles, result)}


def analyze_split(txn, split, keys, rng):
    raw_moments = ExactMoments()
    clipped_moments = ExactMoments()
    channel_moments = [ExactMoments() for _ in STANDARD_32_CHANNELS]
    subject_moments = defaultdict(ExactMoments)
    subject_trial_stds = defaultdict(list)
    label_moments = defaultdict(ExactMoments)
    sampled_values = []
    trial_stds = []
    trial_ptp = []
    trial_abs_max = []
    threshold_counts = {value: 0 for value in (5, 10, 20, 50, 100, 200, 500, 512, 1024)}

    for key in keys:
        pair = pickle.loads(txn.get(key.encode()))
        raw = np.asarray(pair['sample'], dtype=np.float32).reshape(32, -1)
        clipped = np.clip(raw, -CLIP_LIMIT, CLIP_LIMIT)
        raw_moments.update(raw)
        clipped_moments.update(clipped / MODEL_DIVISOR)

        for channel_index in range(raw.shape[0]):
            channel_moments[channel_index].update(raw[channel_index])

        subject = subject_from_key(key)
        subject_moments[subject].update(raw)
        one_trial_std = float(np.std(raw, dtype=np.float64))
        subject_trial_stds[subject].append(one_trial_std)
        label_moments[int(pair['label'])].update(raw)
        trial_stds.append(one_trial_std)
        trial_ptp.append(float(np.ptp(raw)))
        trial_abs_max.append(float(np.max(np.abs(raw))))

        flat = raw.reshape(-1)
        take = min(SAMPLED_VALUES_PER_TRIAL, flat.size)
        sampled_values.append(flat[rng.choice(flat.size, size=take, replace=False)])
        abs_flat = np.abs(flat)
        for threshold in threshold_counts:
            threshold_counts[threshold] += int(np.count_nonzero(abs_flat > threshold))

    sampled_values = np.concatenate(sampled_values).astype(np.float32, copy=False)
    raw_result = raw_moments.result()
    clipped_result = clipped_moments.result()
    total_values = raw_result['count']
    raw_result.update({
        'signed_percentiles_uv': percentiles(sampled_values, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]),
        'absolute_percentiles_uv': percentiles(np.abs(sampled_values), [50, 75, 90, 95, 99, 99.9]),
        'threshold_percentages': {
            str(threshold): count / total_values * 100
            for threshold, count in threshold_counts.items()
        },
        'trial_std_uv': percentiles(trial_stds, [5, 25, 50, 75, 95]),
        'trial_peak_to_peak_uv': percentiles(trial_ptp, [5, 25, 50, 75, 95]),
        'trial_abs_max_uv': percentiles(trial_abs_max, [50, 90, 95, 99]),
    })

    channels = []
    for name, moments in zip(STANDARD_32_CHANNELS, channel_moments):
        row = {'split': split, 'channel': name}
        row.update(moments.result())
        channels.append(row)

    subjects = []
    for subject in sorted(subject_moments):
        row = {'split': split, 'subject': subject, 'trials': len(subject_trial_stds[subject])}
        row.update(subject_moments[subject].result())
        row['median_trial_std'] = float(np.median(subject_trial_stds[subject]))
        subjects.append(row)

    labels = {}
    for label, moments in sorted(label_moments.items()):
        labels[str(label)] = moments.result()

    summary = {
        'split': split,
        'trials': len(keys),
        'raw_uv': raw_result,
        'model_input_after_clip_and_scale': clipped_result,
        'labels': labels,
    }
    return summary, channels, subjects, sampled_values, np.asarray(trial_stds)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path, summaries, channel_rows, subject_rows, sampled_by_split, trial_stds_by_split):
    colors = {'train': '#2f6690', 'val': '#d97706', 'test': '#2f855a'}
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    ax = axes[0, 0]
    for split, values in sampled_by_split.items():
        absolute = np.sort(np.abs(values))
        y = np.linspace(0, 100, absolute.size, endpoint=False)
        ax.plot(absolute, y, label=split, color=colors[split], linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlim(0.02, 1200)
    ax.set_xlabel('|Amplitude| (µV, log scale)')
    ax.set_ylabel('Empirical CDF (%)')
    ax.set_title('Point-wise absolute amplitude')
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    labels = list(trial_stds_by_split)
    ax.boxplot([trial_stds_by_split[name] for name in labels], tick_labels=labels, showfliers=False)
    ax.set_ylabel('Within-trial std (µV)')
    ax.set_title('Trial-level amplitude scale')
    ax.grid(True, axis='y', alpha=0.25)

    ax = axes[1, 0]
    for split in ('train', 'val', 'test'):
        rows = [row for row in subject_rows if row['split'] == split]
        ax.plot([row['subject'] for row in rows], [row['std'] for row in rows], marker='o',
                label=split, color=colors[split])
    ax.axvline(15.5, color='#9aa4af', linestyle='--', linewidth=1)
    ax.axvline(20.5, color='#9aa4af', linestyle='--', linewidth=1)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Global std (µV)')
    ax.set_title('Subject-level amplitude shift')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    x = np.arange(len(STANDARD_32_CHANNELS))
    for split in ('train', 'val', 'test'):
        rows = [row for row in channel_rows if row['split'] == split]
        ax.plot(x, [row['std'] for row in rows], marker='.', label=split, color=colors[split])
    ax.set_xticks(x)
    ax.set_xticklabels(STANDARD_32_CHANNELS, rotation=90, fontsize=7)
    ax.set_ylabel('Channel std (µV)')
    ax.set_title('Per-channel amplitude')
    ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend()

    raw_overall = {row['split']: row['raw_uv']['std'] for row in summaries}
    clipped_overall = {
        row['split']: row['model_input_after_clip_and_scale']['std'] * MODEL_DIVISOR
        for row in summaries
    }
    fig.suptitle(
        'SHU-MI amplitude distribution (raw preprocessed µV)\n'
        f"raw std — {raw_overall['train']:.2f}/{raw_overall['val']:.2f}/{raw_overall['test']:.2f}; "
        f"after ±1024 clip — {clipped_overall['train']:.2f}/{clipped_overall['val']:.2f}/"
        f"{clipped_overall['test']:.2f} µV (train/val/test)",
        fontsize=16,
        weight='bold',
    )
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../BigDownstream/shu_datasets')
    parser.add_argument('--output_dir', default='experiments/reports/shumi_amplitude')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    db = lmdb.open(args.data_dir, readonly=True, lock=False, readahead=True, meminit=False)
    summaries = []
    channel_rows = []
    subject_rows = []
    sampled_by_split = {}
    trial_stds_by_split = {}
    with db.begin(write=False) as txn:
        keys = pickle.loads(txn.get(b'__keys__'))
        for split in ('train', 'val', 'test'):
            summary, channels, subjects, sampled, trial_stds = analyze_split(txn, split, keys[split], rng)
            summaries.append(summary)
            channel_rows.extend(channels)
            subject_rows.extend(subjects)
            sampled_by_split[split] = sampled
            trial_stds_by_split[split] = trial_stds
            print(split, summary['raw_uv'])
    db.close()

    (output_dir / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    write_csv(output_dir / 'channel_stats.csv', channel_rows)
    write_csv(output_dir / 'subject_stats.csv', subject_rows)
    make_plot(
        output_dir / 'shumi_amplitude_distribution.png',
        summaries,
        channel_rows,
        subject_rows,
        sampled_by_split,
        trial_stds_by_split,
    )
    print(output_dir)


if __name__ == '__main__':
    main()
