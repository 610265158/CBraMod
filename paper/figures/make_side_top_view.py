"""Illustrate the side-view (human) versus top-view (CNN) EEG geometry."""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy import signal

from make_synthetic_folding_patterns import fold, make_patterns


INK = "#1D2939"
MUTED = "#667085"
TEAL = "#27788A"
PHASE_COLORS = ("#2E77B5", "#D8752B", "#239B75", "#8558B5")
P = 4
FS = 200


def load_real_channel():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root.parent / (
        "BigDownstream/chb-mit/processed_seg/train/"
        "chb08_13-s-0-add-631295.pkl"
    )
    with path.open("rb") as handle:
        raw = np.asarray(pickle.load(handle)["X"], dtype=np.float32)
    eeg = signal.resample(raw, 2000, axis=1)
    eeg = np.clip(eeg, -1024, 1024) / 32.0
    # Use a five-second crop so the real example has the same horizontal
    # extent as the controlled morphology.  The selected seizure segment has
    # sustained rhythmic activity throughout this crop.
    return eeg[1, : FS * 5]  # F7-T7: clear sustained rhythmic activity.


def side_view(ax, t, y, title, colorize=False):
    ax.plot(t, y, color=INK, linewidth=0.75, zorder=1)
    if colorize:
        # Sparse markers expose the four true polyphase classes without
        # drawing all samples.  Each selected index keeps its original
        # phase (sample index modulo P).
        sample_idx = np.concatenate(
            [np.arange(phase, y.size, P * 8) for phase in range(P)]
        )
        sample_idx.sort()
        ax.scatter(
            t[sample_idx], y[sample_idx], s=7,
            c=[PHASE_COLORS[i % P] for i in sample_idx],
            edgecolors="white", linewidths=0.25, zorder=2,
        )
    ax.axhline(0, color="#D0D5DD", linewidth=0.6)
    ax.set_xlim(0, 5)
    lim = max(np.percentile(np.abs(y), 99.5) * 1.12, 0.3)
    ax.set_ylim(-lim, lim)
    ax.set_title(title, loc="left", fontsize=11.5, pad=6, weight="bold")
    ax.set_xlabel("time (s), side view")
    ax.set_ylabel("amplitude")
    ax.grid(axis="x", color="#EAECF0", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def top_view(ax, y, title):
    folded = fold(y)
    lim = max(np.percentile(np.abs(folded), 99.5), 1e-6)
    ax.imshow(
        np.clip(folded, -lim, lim), aspect="auto", interpolation="nearest",
        cmap="RdBu_r", vmin=-lim, vmax=lim, origin="upper",
    )
    ax.set_xlim(-0.5, folded.shape[1] - 0.5)
    ax.set_ylim(P - 0.5, -0.5)
    ax.set_xticks([0, 50, 100, 150, 200, 249], ["0", "1", "2", "3", "4", "5"])
    ax.set_yticks(range(P), [f"p={p}" for p in range(P)], fontsize=8)
    ax.tick_params(axis="y", pad=5)
    for tick, color in zip(ax.get_yticklabels(), PHASE_COLORS):
        tick.set_color(color)
        tick.set_weight("bold")
    ax.set_xlabel("folded time w (s), top view")
    ax.set_ylabel("phase rows r=p")
    ax.set_title(title, loc="left", fontsize=11.5, pad=6, weight="bold")
    ax.grid(axis="x", color="white", linewidth=0.45, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)


def add_bridge(fig, left_ax, right_ax):
    start = left_ax.get_position().x1 + 0.01
    end = right_ax.get_position().x0 - 0.01
    y = 0.5 * (left_ax.get_position().y0 + left_ax.get_position().y1)
    fig.add_artist(FancyArrowPatch(
        (start, y), (end, y), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=18, linewidth=2.2, color=TEAL,
    ))


def main():
    t, patterns = make_patterns()
    real = load_real_channel()
    rows = [
        (patterns[0], "Synthetic spike–slow-wave"),
        (real, "Real CHB-MIT F7-T7 seizure segment (5 s crop)"),
    ]
    fig = plt.figure(figsize=(15.5, 7.8), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1.0), hspace=0.58, wspace=0.28)
    axes = []
    for row, (y, label) in enumerate(rows):
        side_ax = fig.add_subplot(grid[row, 0])
        top_ax = fig.add_subplot(grid[row, 1])
        side_view(side_ax, t, y, f"{chr(65 + row * 2)}. {label}: side view", colorize=True)
        top_view(top_ax, y, f"{chr(66 + row * 2)}. Same samples: CNN top view after P=4")
        side_ax.text(
            0.01, 0.94, "human-readable waveform",
            transform=side_ax.transAxes, fontsize=8, color=MUTED,
            ha="left", va="top",
        )
        top_ax.text(
            0.99, 0.94, "2D local texture",
            transform=top_ax.transAxes, fontsize=8, color=MUTED,
            ha="right", va="top",
        )
        axes.append((side_ax, top_ax))

    fig.suptitle(
        "One EEG Signal, Two Viewpoints: Side View for Humans, Top View for CNNs",
        y=0.985, fontsize=18, weight="bold", color=INK,
    )
    fig.text(
        0.5, 0.505,
        "Folding is a change of viewpoint and adjacency: no samples are added, removed, or interpolated.",
        ha="center", fontsize=11.2, weight="bold", color="#155462",
    )
    fig.text(
        0.5, 0.47, r"$I[cP+p,w]=X[c,wP+p]$   |   vertical phase neighbors are adjacent to the CNN",
        ha="center", fontsize=10.5, color=INK,
    )
    fig.subplots_adjust(top=0.90, bottom=0.11, left=0.065, right=0.97)
    for side_ax, top_ax in axes:
        add_bridge(fig, side_ax, top_ax)
    output_dir = Path(__file__).resolve().parent
    fig.savefig(output_dir / "side_top_view_folding.png", dpi=300,
                facecolor="white", bbox_inches="tight")
    fig.savefig(output_dir / "side_top_view_folding.pdf",
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
