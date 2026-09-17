"""Combine the synthetic and real folding examples into one paper figure."""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import signal

from make_synthetic_folding_patterns import fold, make_patterns


INK = "#1D2939"
MUTED = "#667085"
TEAL = "#27788A"
PHASE_COLORS = ("#2E77B5", "#D8752B", "#239B75", "#8558B5")
CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)
P = 4
FS = 200


def load_real_sample():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root.parent / (
        "BigDownstream/chb-mit/processed_seg/train/"
        "chb08_13-s-0-add-631295.pkl"
    )
    with path.open("rb") as handle:
        item = pickle.load(handle)
    raw = np.asarray(item["X"], dtype=np.float32)
    if raw.shape != (16, 2560) or int(item["y"]) != 1:
        raise ValueError(f"Unexpected CHB-MIT sample: {raw.shape}, label={item['y']}")
    eeg = signal.resample(raw, 2000, axis=1)
    eeg = np.clip(eeg, -1024.0, 1024.0) / 32.0
    folded = eeg.reshape(16, 500, P).transpose(0, 2, 1).reshape(16 * P, 500)
    return eeg, folded


def raw_panel(ax, t, y, title):
    ax.plot(t, y, color=INK, linewidth=0.7)
    ax.axhline(0, color="#D0D5DD", linewidth=0.5)
    ax.set_xlim(0, 5)
    limit = max(np.percentile(np.abs(y), 99.5) * 1.12, 0.2)
    ax.set_ylim(-limit, limit)
    ax.set_title(title, loc="left", fontsize=8, weight="bold", pad=4)
    ax.grid(axis="x", color="#EAECF0", linewidth=0.45)
    ax.spines[["top", "right"]].set_visible(False)


def folded_panel(ax, values, title, show_y=True):
    limit = max(np.percentile(np.abs(values), 99.5), 1e-6)
    image = ax.imshow(
        np.clip(values, -limit, limit), aspect="auto", interpolation="nearest",
        cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper",
    )
    ax.set_title(title, loc="left", fontsize=8, weight="bold", pad=4)
    ax.set_xticks([])
    if show_y and values.shape[0] == P:
        ax.set_yticks(range(P), [f"p={p}" for p in range(P)], fontsize=7)
        for tick, color in zip(ax.get_yticklabels(), PHASE_COLORS):
            tick.set_color(color)
            tick.set_weight("bold")
    else:
        ax.set_yticks([])
    return image


def stacked_real(ax, eeg):
    seconds = np.arange(eeg.shape[1]) / FS
    offsets = np.arange(15, -1, -1, dtype=float)
    gain = 0.40 / max(np.percentile(np.abs(eeg), 99.5), 1e-6)
    for index, offset in enumerate(offsets):
        ax.plot(seconds, offset + eeg[index] * gain, color=INK, linewidth=0.48)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.7, 15.7)
    ax.set_yticks(offsets, CHANNELS, fontsize=7)
    ax.set_title("G. CHB-MIT: 16 channels", loc="left",
                 fontsize=8, weight="bold", pad=4)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.grid(axis="x", color="#EAECF0", linewidth=0.45)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    # Design at the final ICLR text width, so point sizes survive inclusion.
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42})
    t, patterns = make_patterns()
    eeg, real_folded = load_real_sample()
    names = ("Spike--slow-wave", "Rhythmic oscillation", "Periodic discharges")

    fig = plt.figure(figsize=(5.39, 6.5), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.15), hspace=0.32)
    synthetic = outer[0].subgridspec(3, 2, width_ratios=(1.25, 1.0),
                                     hspace=0.55, wspace=0.30)
    for row, (name, values) in enumerate(zip(names, patterns)):
        raw_ax = fig.add_subplot(synthetic[row, 0])
        fold_ax = fig.add_subplot(synthetic[row, 1])
        raw_panel(raw_ax, t, values, f"{chr(65 + 2 * row)}. {name}")
        folded_panel(fold_ax, fold(values),
                     f"{chr(66 + 2 * row)}. Folded (P=4)")
        if row < 2:
            raw_ax.set_xticklabels([])
        else:
            raw_ax.set_xlabel("Time (s)", fontsize=7)
            fold_ax.set_xticks([0, 125, 249], ["0", "2.5", "5"])
            fold_ax.set_xlabel("Time (s); column step = 20 ms", fontsize=7)

    real = outer[1].subgridspec(2, 2, width_ratios=(1.25, 1.0),
                               height_ratios=(0.65, 1.35), wspace=0.30, hspace=0.58)
    stack_ax = fig.add_subplot(real[:, 0])
    channel_ax = fig.add_subplot(real[0, 1])
    tensor_ax = fig.add_subplot(real[1, 1])
    stacked_real(stack_ax, eeg)

    seconds = np.arange(eeg.shape[1]) / FS
    channel_ax.plot(seconds, eeg[1], color=TEAL, linewidth=0.65)
    channel_ax.set_xlim(0, 10)
    channel_ax.set_title("H. F7–T7 detail", loc="left",
                         fontsize=8, weight="bold", pad=4)
    channel_ax.set_xlabel("Time (s)", fontsize=7)
    channel_ax.grid(color="#EAECF0", linewidth=0.45)
    channel_ax.spines[["top", "right"]].set_visible(False)

    folded_panel(tensor_ax, real_folded,
                 "I. Folded tensor (64 × 500)",
                 show_y=False)
    for boundary in range(P, real_folded.shape[0], P):
        tensor_ax.axhline(boundary - 0.5, color="white", linewidth=0.45)
    for row in range(real_folded.shape[0]):
        tensor_ax.add_patch(Rectangle(
            (-15, row - 0.5), 10, 1, facecolor=PHASE_COLORS[row % P],
            edgecolor="none", clip_on=False,
        ))
    tensor_ax.set_xticks([0, 250, 499], ["0", "5", "10"])
    tensor_ax.set_xlabel("Time (s); column step = 20 ms", fontsize=7)
    fig.subplots_adjust(top=0.965, bottom=0.07, left=0.13, right=0.98)

    output = Path(__file__).resolve().parent
    fig.savefig(output / "combined_folding_examples.png", dpi=300,
                facecolor="white")
    fig.savefig(output / "combined_folding_examples.pdf",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
