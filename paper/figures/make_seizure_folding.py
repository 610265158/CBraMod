"""Generate a real CHB-MIT seizure example for the folding section."""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import signal


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
SAMPLE = "chb08_13-s-0-add-631295.pkl"
SPLIT = "train"
FOLD_FACTOR = 4
RAW_RATE = 256
MODEL_POINTS = 2000


def load_sample():
    repo_root = Path(__file__).resolve().parents[2]
    sample_path = repo_root.parent / "BigDownstream/chb-mit/processed_seg" / SPLIT / SAMPLE
    with sample_path.open("rb") as handle:
        item = pickle.load(handle)
    raw = np.asarray(item["X"], dtype=np.float32)
    label = int(item["y"])
    if raw.shape != (16, 2560) or label != 1:
        raise ValueError(f"Unexpected CHB sample: shape={raw.shape}, label={label}")

    # Match the downstream loader: resample to 2,000 points, clip, then /32.
    eeg = signal.resample(raw, MODEL_POINTS, axis=1)
    eeg = np.clip(eeg, -1024.0, 1024.0) / 32.0
    width = MODEL_POINTS // FOLD_FACTOR
    folded = (
        eeg.reshape(eeg.shape[0], width, FOLD_FACTOR)
        .transpose(0, 2, 1)
        .reshape(eeg.shape[0] * FOLD_FACTOR, width)
    )
    unfolded = (
        folded.reshape(eeg.shape[0], FOLD_FACTOR, width)
        .transpose(0, 2, 1)
        .reshape(eeg.shape)
    )
    if not np.array_equal(eeg, unfolded):
        raise RuntimeError("CHB fold/unfold verification failed")
    return eeg, folded, label


def stacked_wave(ax, eeg, rate, title, channels=True):
    seconds = np.arange(eeg.shape[1]) / rate
    offsets = np.arange(eeg.shape[0] - 1, -1, -1, dtype=float)
    gain = 0.40 / max(np.percentile(np.abs(eeg), 99.5), 1e-6)
    for index, offset in enumerate(offsets):
        ax.plot(seconds, offset + eeg[index] * gain, color=INK, lw=0.55, alpha=0.9)
    ax.set_title(title, loc="left", pad=7, fontsize=12.5, weight="bold")
    ax.set_xlim(seconds[0], seconds[-1])
    ax.set_ylim(-0.7, eeg.shape[0] - 0.3)
    ax.set_xlabel(f"time (s), {rate:g} Hz")
    if channels:
        ax.set_yticks(offsets, CHANNELS, fontsize=7.0)
    else:
        ax.set_yticks([])
    ax.grid(axis="x", color="#EAECF0", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    eeg, folded, label = load_sample()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.edgecolor": "#98A2B3",
    })
    fig = plt.figure(figsize=(15.5, 9.8), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.18), hspace=0.42)
    top = outer[0].subgridspec(1, 2, width_ratios=(1.4, 1.0), wspace=0.18)
    bottom = outer[1].subgridspec(1, 2, width_ratios=(1.0, 1.42), wspace=0.18)

    raw_ax = fig.add_subplot(top[0, 0])
    stacked_wave(raw_ax, eeg, MODEL_POINTS / 10.0,
                 "A. Real CHB-MIT seizure segment")
    raw_ax.text(
        0.01, 0.99,
        f"{SPLIT}/{SAMPLE}   label={label}: seizure\n"
        "same preprocessing as model: resample 2560→2000, clip ±1024, divide by 32",
        transform=raw_ax.transAxes, va="top", ha="left", fontsize=7.5, color=MUTED,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2),
    )

    # F7-T7 shows the sustained rhythmic activity in the labeled ictal window
    # without letting the isolated onset spike dominate the zoom scale.
    channel = 1
    zoom_ax = fig.add_subplot(top[0, 1])
    seconds = np.arange(eeg.shape[1]) / (MODEL_POINTS / 10.0)
    zoom_ax.plot(seconds, eeg[channel], color=TEAL, lw=0.75)
    zoom_ax.axvspan(0.5, 9.8, color="#D8752B", alpha=0.10,
                    label="sustained rhythmic ictal interval")
    zoom_ax.set_title(
        f"B. Enlarged channel: {CHANNELS[channel]}",
        loc="left", pad=7, fontsize=12.5, weight="bold",
    )
    zoom_ax.set_xlim(0, 10)
    zoom_ax.set_xlabel("time (s), model input")
    zoom_ax.set_ylabel("amplitude (/32)")
    zoom_ax.grid(color="#EAECF0", lw=0.55)
    zoom_ax.legend(frameon=False, fontsize=8, loc="upper right")
    zoom_ax.spines[["top", "right"]].set_visible(False)
    zoom_ax.text(
        0.02, 0.04,
        "sustained rhythmic activity is visible across multiple channels",
        transform=zoom_ax.transAxes, fontsize=8, color=MUTED,
    )

    heat_ax = fig.add_subplot(bottom[0, 0])
    limit = max(np.percentile(np.abs(folded), 99.5), 1e-6)
    heat = heat_ax.imshow(
        np.clip(folded, -limit, limit), aspect="auto", interpolation="nearest",
        cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper",
    )
    for boundary in range(FOLD_FACTOR, folded.shape[0], FOLD_FACTOR):
        heat_ax.axhline(boundary - 0.5, color="white", lw=0.7)
    for row in range(folded.shape[0]):
        heat_ax.add_patch(Rectangle(
            (-15, row - 0.5), 10, 1, facecolor=PHASE_COLORS[row % FOLD_FACTOR],
            edgecolor="none", clip_on=False,
        ))
    heat_ax.set_title("C. Lossless folded seizure tensor: 64×500", loc="left",
                      pad=7, fontsize=12.5, weight="bold")
    heat_ax.set_xlim(-0.5, folded.shape[1] - 0.5)
    heat_ax.set_xticks([0, 100, 200, 300, 400, 499], ["0", "1", "2", "3", "4", "5"])
    heat_ax.set_xlabel("folded time (s); one column = 20 ms")
    heat_ax.set_yticks([c * FOLD_FACTOR + 1.5 for c in range(16)],
                       [f"{c + 1}" for c in range(16)], fontsize=6)
    heat_ax.set_ylabel("channel groups × 4 phase rows", fontsize=8)
    heat_ax.tick_params(length=2, pad=1)
    colorbar = fig.colorbar(heat, ax=heat_ax, fraction=0.025, pad=0.02)
    colorbar.set_label("amplitude (/32)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    phase_ax = fig.add_subplot(bottom[0, 1])
    width = folded.shape[1]
    rows = folded[channel * FOLD_FACTOR:(channel + 1) * FOLD_FACTOR]
    offsets = np.arange(FOLD_FACTOR - 1, -1, -1, dtype=float)
    phase_gain = 0.34 / max(np.percentile(np.abs(rows), 99.5), 1e-6)
    folded_seconds = np.arange(width) / 100.0
    for phase in range(FOLD_FACTOR):
        base = offsets[phase]
        phase_ax.axhspan(base - 0.42, base + 0.42, color=PHASE_COLORS[phase], alpha=0.09)
        phase_ax.axhline(base, color="#D0D5DD", lw=0.5)
        phase_ax.plot(
            folded_seconds, base + rows[phase] * phase_gain,
            color=PHASE_COLORS[phase], lw=0.85,
        )
    phase_ax.set_title(
        f"D. {CHANNELS[channel]} after folding: four adjacent phase rows",
        loc="left", pad=7, fontsize=12.5, weight="bold",
    )
    phase_ax.set_xlim(0, 5)
    phase_ax.set_ylim(-0.48, 3.48)
    phase_ax.set_xlabel("folded time (s); horizontal step = P=4 samples")
    phase_ax.set_yticks(offsets, [f"p={p}" for p in range(FOLD_FACTOR)])
    for tick, color in zip(phase_ax.get_yticklabels(), PHASE_COLORS):
        tick.set_color(color)
        tick.set_weight("bold")
    phase_ax.grid(axis="x", color="#EAECF0", lw=0.55)
    phase_ax.spines[["top", "right"]].set_visible(False)
    phase_ax.text(
        0.995, 0.03, r"$I[cP+p,w]=X[c,wP+p]$",
        transform=phase_ax.transAxes, ha="right", va="bottom", fontsize=11, color=INK,
    )

    fig.suptitle(
        "Lossless Temporal Folding Reveals Seizure Activity as Local CNN Texture",
        y=0.988, fontsize=18, weight="bold", color=INK,
    )
    fig.text(
        0.5, 0.505,
        "Real CHB-MIT seizure example  ·  same 32,000 values  ·  only adjacency changes",
        ha="center", fontsize=11.5, weight="bold", color="#155462",
    )
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.055, right=0.975)
    output_dir = Path(__file__).resolve().parent
    fig.savefig(output_dir / "seizure_lossless_folding.png", dpi=300,
                facecolor="white", bbox_inches="tight")
    fig.savefig(output_dir / "seizure_lossless_folding.pdf",
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
