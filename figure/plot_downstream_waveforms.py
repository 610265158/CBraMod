#!/usr/bin/env python
"""Plot representative downstream EEG samples in an MNE-browser-like style."""

import argparse
import csv
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.downstream import DOWNSTREAM_11_CONFIGS
from datasets.channel_mirror import DATASET_CHANNEL_ORDERS


SAMPLING_RATES = {name: 200 for name in DOWNSTREAM_11_CONFIGS}
MODEL_SCALE_DIVISORS = {name: 32.0 for name in DOWNSTREAM_11_CONFIGS}
MODEL_SCALE_DIVISORS['SHU-MI'] = 1.0
OVERVIEW_CHANNELS = 10
REPRESENTATIVE_CANDIDATES = 24


def loader_params(name, config):
    return SimpleNamespace(
        datasets_dir=config['datasets_dir'],
        batch_size=1,
        num_workers=0,
        device='cpu',
        seed=3407,
        downstream_dataset=name,
        downstream_task=config['task'],
        balanced_sampling=False,
        mirror_augmentation=False,
        mirror_prob=0.5,
        time_roll_augmentation=False,
        time_roll_prob=1.0,
        time_roll_max_fraction=0.5,
        amplitude_scale_augmentation=False,
        amplitude_scale_prob=0.5,
        amplitude_scale_min=0.25,
        amplitude_scale_max=1.25,
    )


def representative_sample(name, config):
    module = import_module(config['dataset_module'])
    loaders = module.LoadDataset(loader_params(name, config)).get_data_loader()
    candidates = []

    for batch_index, (x, y) in enumerate(loaders['val']):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        if x.ndim == 4:  # ISRUC: [B, 20, C, T]
            for epoch_index in range(x.shape[1]):
                signal = x[0, epoch_index]
                candidates.append((signal, y[0, epoch_index], f'val sequence {batch_index}, epoch {epoch_index}'))
                if len(candidates) >= REPRESENTATIVE_CANDIDATES:
                    break
        else:
            candidates.append((x[0], y.reshape(-1)[0], f'val sample {batch_index}'))
        if len(candidates) >= REPRESENTATIVE_CANDIDATES:
            break

    if not candidates:
        raise RuntimeError(f'No validation samples found for {name}')

    standard_deviations = np.asarray([np.std(item[0]) for item in candidates])
    median_std = np.median(standard_deviations)
    selected = int(np.argmin(np.abs(standard_deviations - median_std)))
    signal, label, source = candidates[selected]
    return np.asarray(signal, dtype=np.float32), label, source


def channel_names(name, count):
    configured = DATASET_CHANNEL_ORDERS.get(name, ())
    if len(configured) == count:
        return list(configured)
    return [f'CH{index + 1:02d}' for index in range(count)]


def evenly_spaced_indices(count, limit):
    if count <= limit:
        return np.arange(count)
    return np.unique(np.rint(np.linspace(0, count - 1, limit)).astype(int))


def robust_scale(centered):
    scale = float(np.percentile(np.abs(centered), 98))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(centered))
    return max(scale, 1e-6)


def plot_browser(ax, signal, names, sfreq, title, scale_divisor, max_channels=None):
    indices = evenly_spaced_indices(signal.shape[0], max_channels or signal.shape[0])
    shown = signal[indices] * scale_divisor
    shown_names = [names[index] for index in indices]
    shown = shown - shown.mean(axis=1, keepdims=True)
    scale_uv = robust_scale(shown)
    offsets = np.arange(len(indices) - 1, -1, -1, dtype=np.float32)
    time = np.arange(shown.shape[1], dtype=np.float32) / sfreq

    ax.set_facecolor('#fbfbfc')
    for row, offset in zip(shown, offsets):
        ax.plot(time, offset + 0.34 * row / scale_uv, color='#244b74', linewidth=0.55)
        ax.axhline(offset, color='#d7dce2', linewidth=0.35, zorder=0)

    duration = time[-1] if len(time) else 0
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.8, len(indices) - 0.2)
    ax.set_yticks(offsets)
    ax.set_yticklabels(shown_names, fontsize=7)
    ax.tick_params(axis='y', length=0, pad=3)
    ax.tick_params(axis='x', labelsize=7)
    ax.xaxis.grid(True, color='#d7dce2', linewidth=0.45)
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_title(title, loc='left', fontsize=10, weight='bold', pad=5)

    bar_x = duration * 0.965
    bar_y = -0.48
    ax.plot([bar_x, bar_x], [bar_y, bar_y + 0.34], color='#b33b32', linewidth=1.5)
    ax.text(bar_x, bar_y + 0.39, f'{scale_uv:.1f} µV', color='#8f302a', fontsize=6,
            ha='center', va='bottom')

    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color('#9aa4af')
    ax.spines['bottom'].set_color('#9aa4af')
    return scale_uv, indices


def safe_name(name):
    return name.lower().replace('-', '_').replace(' ', '_')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='figure/downstream_waveforms_mne')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = {}
    metadata = []
    for name, config in DOWNSTREAM_11_CONFIGS.items():
        signal, label, source = representative_sample(name, config)
        names = channel_names(name, signal.shape[0])
        samples[name] = (signal, label, source, names)

    fig, axes = plt.subplots(4, 3, figsize=(22, 27), constrained_layout=True)
    axes = axes.reshape(-1)
    for ax, (name, config) in zip(axes, DOWNSTREAM_11_CONFIGS.items()):
        signal, label, source, names = samples[name]
        raw_std = float(np.std(signal) * MODEL_SCALE_DIVISORS[name])
        title = f'{name}   shape={tuple(signal.shape)}   label={label}   σ={raw_std:.2f} µV'
        scale_uv, shown_indices = plot_browser(
            ax, signal, names, SAMPLING_RATES[name], title, MODEL_SCALE_DIVISORS[name],
            max_channels=OVERVIEW_CHANNELS,
        )
        metadata.append({
            'dataset': name,
            'source': source,
            'label': label,
            'shape': str(tuple(signal.shape)),
            'sampling_rate_hz': SAMPLING_RATES[name],
            'duration_s': signal.shape[-1] / SAMPLING_RATES[name],
            'model_input_std': float(np.std(signal)),
            'display_std_uv': raw_std,
            'display_scale_uv': scale_uv,
            'overview_channels': ','.join(names[index] for index in shown_indices),
        })

    for ax in axes[len(DOWNSTREAM_11_CONFIGS):]:
        ax.axis('off')
    fig.suptitle(
        'CBraMod downstream EEG waveforms — MNE-style overview\n'
        'Validation samples after dataset preprocessing; display converts model input back by ×32. '
        'Each panel is independently autoscaled.',
        fontsize=18,
        weight='bold',
    )
    overview_path = output_dir / 'all_datasets_mne_overview.png'
    fig.savefig(overview_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    for name in DOWNSTREAM_11_CONFIGS:
        signal, label, source, names = samples[name]
        raw_std = float(np.std(signal) * MODEL_SCALE_DIVISORS[name])
        height = max(8.5, 3.0 + signal.shape[0] * 0.28)
        fig, ax = plt.subplots(figsize=(19, height), constrained_layout=True)
        title = (
            f'{name} — all {signal.shape[0]} channels   shape={tuple(signal.shape)}   '
            f'label={label}   σ={raw_std:.2f} µV   {source}'
        )
        plot_browser(ax, signal, names, SAMPLING_RATES[name], title, MODEL_SCALE_DIVISORS[name])
        fig.savefig(output_dir / f'{safe_name(name)}_all_channels.png', dpi=180, bbox_inches='tight')
        plt.close(fig)

    with (output_dir / 'sample_metadata.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    print(overview_path)
    print(output_dir / 'sample_metadata.csv')


if __name__ == '__main__':
    main()
