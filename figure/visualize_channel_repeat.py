#!/usr/bin/env python3
"""Visualize consecutive EEG channel repetition on a real FACED sample."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.faced_dataset import CustomDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="../BigDownstream/faced/processed",
        help="FACED LMDB directory",
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument(
        "--output",
        default="figure/channel_repeat_faced_sample.png",
    )
    parser.add_argument("--zoom_channels", type=int, default=4)
    parser.add_argument("--zoom_samples", type=int, default=800)
    return parser.parse_args()


def repeated(eeg, factor):
    return np.repeat(eeg, repeats=factor, axis=0)


def add_group_boundaries(axis, groups, factor, color="white"):
    for group in range(1, groups):
        axis.axhline(group * factor - 0.5, color=color, linewidth=0.7, alpha=0.9)


def main():
    args = parse_args()
    dataset = CustomDataset(args.data_dir, mode=args.split)
    eeg, label = dataset[args.sample_index]
    eeg = np.asarray(eeg, dtype=np.float32)

    factors = (1, 2, 4)
    matrices = [repeated(eeg, factor) for factor in factors]
    color_limit = float(np.percentile(np.abs(eeg), 99.5))
    zoom_channels = min(args.zoom_channels, eeg.shape[0])
    zoom_samples = min(args.zoom_samples, eeg.shape[1])

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 8.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 0.9)},
    )

    last_image = None
    for column, (factor, matrix) in enumerate(zip(factors, matrices)):
        last_image = axes[0, column].imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
        )
        axes[0, column].set_title(
            f"repeat={factor}: {matrix.shape[0]} channels x {matrix.shape[1]} samples"
        )
        axes[0, column].set_xlabel("Time sample")
        axes[0, column].set_ylabel("Input row")

        zoom = matrix[: zoom_channels * factor, :zoom_samples]
        axes[1, column].imshow(
            zoom,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
        )
        add_group_boundaries(axes[1, column], zoom_channels, factor)
        axes[1, column].set_title(
            f"Zoom: {zoom_channels} source channels x {factor} adjacent row"
            f"{'s' if factor > 1 else ''}"
        )
        axes[1, column].set_xlabel("Time sample")
        axes[1, column].set_ylabel("Repeated input row")

        tick_positions = [group * factor + (factor - 1) / 2 for group in range(zoom_channels)]
        axes[1, column].set_yticks(tick_positions)
        axes[1, column].set_yticklabels(
            [f"source ch {group + 1}" for group in range(zoom_channels)]
        )

    figure.suptitle(
        "Consecutive channel repeat on one real FACED EEG sample "
        f"(split={args.split}, index={args.sample_index}, label={int(label)})",
        fontsize=15,
    )
    colorbar = figure.colorbar(last_image, ax=axes, shrink=0.86, pad=0.015)
    colorbar.set_label("Normalized EEG amplitude (clip to [-1024, 1024], then /32)")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(output.resolve())


if __name__ == "__main__":
    main()
