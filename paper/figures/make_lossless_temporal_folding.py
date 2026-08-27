"""Generate the publication figure for phase-interleaved temporal folding."""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


COLORS = ("#2E77B5", "#D8752B", "#239B75", "#8558B5")
INK = "#1D2939"
MUTED = "#667085"
TEAL = "#27788A"

TUEV_SAMPLE = "aaaaacpc_00000001-66.pkl"
TUEV_LABELS = {
    1: "SPSW",
    2: "GPED",
    3: "PLED",
    4: "EYEM",
    5: "ARTF",
    6: "BCKG",
}
TUEV_CHANNELS = (
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)


def load_tuev_example(fold_factor=4):
    repo_root = Path(__file__).resolve().parents[2]
    sample_path = (
        repo_root.parent
        / "BigDownstream/TUEV_refine/processed/processed_train"
        / TUEV_SAMPLE
    )
    with sample_path.open("rb") as handle:
        sample = pickle.load(handle)

    raw = np.asarray(sample["signal"], dtype=np.float32)
    label = int(np.asarray(sample["label"]).reshape(-1)[0])
    if raw.shape != (16, 1000):
        raise ValueError(f"Expected a 16x1000 TUEV example, got {raw.shape}")

    # This exactly matches datasets.shape_utils.clip_eeg for regular datasets.
    eeg = np.clip(raw, -1024.0, 1024.0) / 32.0
    channels, time_points = eeg.shape
    if time_points % fold_factor:
        raise ValueError("The TUEV time dimension must be divisible by P")
    width = time_points // fold_factor
    folded = (
        eeg.reshape(channels, width, fold_factor)
        .transpose(0, 2, 1)
        .reshape(channels * fold_factor, width)
    )
    unfolded = (
        folded.reshape(channels, fold_factor, width)
        .transpose(0, 2, 1)
        .reshape(channels, time_points)
    )
    if not np.array_equal(unfolded, eeg):
        raise RuntimeError("Fold/unfold verification failed")
    return eeg, folded, label


def draw_concept_panels(fig, top_grid, fold_factor=4):
    samples = np.array([
        -0.05, -0.42, 0.45, 0.18,
        0.72, -0.18, 0.34, -0.58,
        0.06, 0.56, -0.31, 0.25,
        0.91, 0.42, -0.12, 0.61,
    ])
    time = np.arange(samples.size)
    width = samples.size // fold_factor
    reshaped_indices = time.reshape(width, fold_factor)
    folded_indices = time.reshape(width, fold_factor).T
    folded_values = samples.reshape(width, fold_factor).T

    raw_ax = fig.add_subplot(top_grid[0, 0])
    map_ax = fig.add_subplot(top_grid[0, 1])
    fold_ax = fig.add_subplot(top_grid[0, 2])

    raw_ax.set_title("A. Raw EEG channel", pad=13, fontsize=14)
    raw_ax.plot(time, samples, color=INK, linewidth=1.8, zorder=1)
    for phase in range(fold_factor):
        selected = time % fold_factor == phase
        raw_ax.scatter(
            time[selected], samples[selected], s=58, color=COLORS[phase],
            edgecolor="white", linewidth=1.0, zorder=3, label=f"phase {phase}",
        )
    for index, value in zip(time, samples):
        raw_ax.text(
            index, value + 0.13, str(index), ha="center", va="bottom",
            fontsize=8, color=COLORS[index % fold_factor], weight="bold",
        )
    raw_ax.axhline(0, color="#D0D5DD", linewidth=0.8)
    raw_ax.set_xlim(-0.7, 15.7)
    raw_ax.set_ylim(-0.82, 1.17)
    raw_ax.set_xlabel("time index t")
    raw_ax.set_ylabel("amplitude")
    raw_ax.set_xticks(range(16))
    raw_ax.set_yticks([])
    raw_ax.grid(axis="x", color="#EAECF0", linewidth=0.6)
    raw_ax.legend(ncol=2, frameon=False, loc="lower left", fontsize=8.5)
    raw_ax.text(
        0.5, -0.18, "colors repeat every P=4 samples:  p = t mod P",
        transform=raw_ax.transAxes, ha="center", color=MUTED,
    )

    map_ax.set_title("B. Reindex", pad=13, fontsize=14)
    map_ax.set_axis_off()
    table = map_ax.table(
        cellText=reshaped_indices,
        cellLoc="center",
        loc="upper center",
        bbox=(0.12, 0.43, 0.76, 0.40),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(1.0)
        cell.set_facecolor("#F8FAFC")
        cell.get_text().set_color(COLORS[column])
        cell.get_text().set_weight("bold")
    map_ax.text(0.5, 0.88, "reshape to (W, P)", ha="center", color=MUTED)
    map_ax.text(0.5, 0.36, "transpose → (P, W)", ha="center", color=INK, weight="bold")
    map_ax.annotate(
        "", xy=(0.5, 0.39), xytext=(0.5, 0.27),
        arrowprops=dict(arrowstyle="<->", color=TEAL, lw=2.2),
    )
    map_ax.text(0.5, 0.22, "Fold ↔ Unfold", ha="center", color=TEAL, weight="bold")
    map_ax.text(0.5, 0.16, "permutation only\nno interpolation", ha="center", color=MUTED)

    fold_ax.set_title("C. Folded EEG rows", pad=13, fontsize=14)
    row_gap = 1.25
    for phase in range(fold_factor):
        baseline = (fold_factor - 1 - phase) * row_gap
        values = folded_values[phase]
        indices = folded_indices[phase]
        normalized = 0.46 * values + baseline
        fold_ax.axhspan(
            baseline - 0.48, baseline + 0.48,
            color=COLORS[phase], alpha=0.10, zorder=0,
        )
        fold_ax.plot(np.arange(width), normalized, color=INK, linewidth=1.7, zorder=1)
        fold_ax.scatter(
            np.arange(width), normalized, s=62, color=COLORS[phase],
            edgecolor="white", linewidth=1.0, zorder=3,
        )
        for column, (index, value) in enumerate(zip(indices, normalized)):
            fold_ax.text(
                column, value + 0.16, str(index), ha="center", va="bottom",
                fontsize=9, color=COLORS[phase], weight="bold",
            )
        fold_ax.text(
            -0.62, baseline, f"p={phase}", ha="right", va="center",
            color=COLORS[phase], weight="bold",
        )
    fold_ax.set_xlim(-1.0, width - 0.45)
    fold_ax.set_ylim(-0.72, fold_factor * row_gap - 0.18)
    fold_ax.set_xticks(range(width), [f"w={value}" for value in range(width)])
    fold_ax.set_yticks([])
    fold_ax.set_xlabel("horizontal step Δw=1 corresponds to Δt=P")
    fold_ax.grid(axis="x", color="#EAECF0", linewidth=0.7)
    fold_ax.add_patch(
        Rectangle(
            (-0.28, 0.78), 2.58, 3.48, fill=False,
            edgecolor="#C43D3D", linewidth=1.8, linestyle="--",
        )
    )
    fold_ax.text(
        2.5, 3.98, "local CNN\nreceptive field", color="#A52F2F",
        fontsize=9, ha="right",
    )
    fold_ax.text(
        0.5, -0.18, r"$I[cP+p,w]=X[c,wP+p]$",
        transform=fold_ax.transAxes, ha="center", fontsize=13, color=INK,
    )
    return raw_ax, map_ax, fold_ax


def draw_tuev_panel(fig, bottom_grid, eeg, folded, label, fold_factor=4):
    wave_ax = fig.add_subplot(bottom_grid[0, 0])
    arrow_ax = fig.add_subplot(bottom_grid[0, 1])
    folded_grid = bottom_grid[0, 2].subgridspec(
        2, 1, height_ratios=(0.78, 0.92), hspace=0.40,
    )
    heat_ax = fig.add_subplot(folded_grid[0, 0])
    phase_ax = fig.add_subplot(folded_grid[1, 0])

    seconds = np.arange(eeg.shape[1]) / 200.0
    offsets = np.arange(eeg.shape[0] - 1, -1, -1, dtype=float)
    robust_peak = np.percentile(np.abs(eeg), 99.5)
    display_gain = 0.40 / max(robust_peak, 1e-6)
    for channel, offset in enumerate(offsets):
        wave_ax.plot(
            seconds, offset + eeg[channel] * display_gain,
            color=INK, linewidth=0.55, alpha=0.92,
        )
    wave_ax.set_title("D1. Real TUEV EEG segment", loc="left", pad=9, fontsize=12.5)
    wave_ax.set_xlim(0, 5)
    wave_ax.set_ylim(-0.7, 15.7)
    wave_ax.set_xlabel("time (s), 200 Hz")
    wave_ax.set_yticks(offsets, TUEV_CHANNELS, fontsize=7.2)
    wave_ax.grid(axis="x", color="#EAECF0", linewidth=0.55)
    wave_ax.spines[["top", "right"]].set_visible(False)
    wave_ax.text(
        0.01, 0.99,
        f"train/{TUEV_SAMPLE}   label {label}: {TUEV_LABELS.get(label, 'unknown')}\n"
        "pipeline input: clip ±1024 μV, divide by 32; shared display gain",
        transform=wave_ax.transAxes, va="top", ha="left", fontsize=7.8,
        color=MUTED, bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=2),
    )

    arrow_ax.set_axis_off()
    arrow_ax.annotate(
        "", xy=(0.92, 0.58), xytext=(0.08, 0.58),
        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.5, mutation_scale=18),
    )
    arrow_ax.text(
        0.5, 0.72, "phase-interleaved\nfold, P=4",
        ha="center", va="center", fontsize=9.5, weight="bold", color=TEAL,
    )
    arrow_ax.text(
        0.5, 0.43, "16×1000\n→\n64×250",
        ha="center", va="center", fontsize=10.5, weight="bold", color=INK,
    )
    arrow_ax.text(
        0.5, 0.18, "same 16,000 values",
        ha="center", va="center", fontsize=8.2, color=MUTED,
    )

    # A front-facing tensor view makes the repeated p=0,1,2,3 row pattern
    # explicit.  The amplitude color scale is shared across the full tensor.
    amplitude_limit = max(np.percentile(np.abs(folded), 99.5), 1e-6)
    heat = heat_ax.imshow(
        np.clip(folded, -amplitude_limit, amplitude_limit),
        aspect="auto", interpolation="nearest", cmap="RdBu_r",
        vmin=-amplitude_limit, vmax=amplitude_limit, origin="upper",
    )
    for boundary in range(fold_factor, folded.shape[0], fold_factor):
        heat_ax.axhline(boundary - 0.5, color="white", linewidth=0.75, alpha=0.95)
    for row in range(folded.shape[0]):
        phase = row % fold_factor
        heat_ax.add_patch(Rectangle(
            (-8.5, row - 0.5), 6.0, 1.0, facecolor=COLORS[phase],
            edgecolor="none", clip_on=False,
        ))
    heat_ax.set_title(
        "D2. Folded tensor: 64 phase rows × 250 steps",
        loc="left", pad=5, fontsize=11.5,
    )
    heat_ax.set_xlim(-0.5, folded.shape[1] - 0.5)
    heat_ax.set_xticks([0, 50, 100, 150, 200, 249],
                       ["0", "1", "2", "3", "4", "5"])
    heat_ax.set_xlabel("folded time (s); one column = P/200 = 20 ms", labelpad=2)
    shown_channels = [0, 3, 7, 11, 15]
    heat_ax.set_yticks(
        [channel * fold_factor + 1.5 for channel in shown_channels],
        [f"ch {channel + 1}" for channel in shown_channels], fontsize=6.2,
    )
    heat_ax.set_ylabel("16 electrodes × 4 phases", labelpad=2, fontsize=7.5)
    heat_ax.tick_params(axis="both", length=2, pad=1)
    colorbar = fig.colorbar(heat, ax=heat_ax, fraction=0.020, pad=0.015)
    colorbar.set_label("amplitude (/32)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, length=2)

    heat_ax.text(
        0.995, 0.98, "white lines: electrode boundaries\nleft strip: p=0,1,2,3",
        transform=heat_ax.transAxes, ha="right", va="top", fontsize=6.5,
        color=INK, bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
    )

    # A single real electrode is enlarged below.  The four parallel traces
    # expose the phase-row pattern without the occlusion caused by 3D viewing.
    zoom_channel = 10  # C3-P3: a visibly active channel in this sample.
    zoom_rows = folded[
        zoom_channel * fold_factor:(zoom_channel + 1) * fold_factor
    ]
    zoom_limit = max(np.percentile(np.abs(zoom_rows), 99.5), 1e-6)
    folded_seconds = np.arange(folded.shape[1]) * fold_factor / 200.0
    phase_offsets = np.arange(fold_factor - 1, -1, -1, dtype=float)
    zoom_gain = 0.36 / zoom_limit
    for phase in range(fold_factor):
        baseline = phase_offsets[phase]
        phase_ax.axhspan(
            baseline - 0.43, baseline + 0.43,
            color=COLORS[phase], alpha=0.09, zorder=0,
        )
        phase_ax.axhline(baseline, color="#D0D5DD", linewidth=0.55, zorder=1)
        phase_ax.plot(
            folded_seconds,
            baseline + np.clip(zoom_rows[phase], -zoom_limit, zoom_limit) * zoom_gain,
            color=COLORS[phase], linewidth=0.90, zorder=2,
        )
    phase_ax.set_title(
        f"D3. Real {TUEV_CHANNELS[zoom_channel]} enlarged: four adjacent phase rows",
        loc="left", pad=7, fontsize=10.2,
    )
    phase_ax.set_xlim(0, 5)
    phase_ax.set_ylim(-0.48, 3.48)
    phase_ax.set_xlabel("folded time (s); adjacent points are P=4 samples apart")
    phase_ax.set_yticks(
        phase_offsets, [f"p={phase}" for phase in range(fold_factor)], fontsize=7.5,
    )
    for tick, color in zip(phase_ax.get_yticklabels(), COLORS):
        tick.set_color(color)
        tick.set_weight("bold")
    phase_ax.grid(axis="x", color="#EAECF0", linewidth=0.55)
    phase_ax.spines[["top", "right"]].set_visible(False)
    phase_ax.text(
        0.995, 0.02, r"$I[cP+p,w]=X[c,wP+p]$",
        transform=phase_ax.transAxes, ha="right", va="bottom",
        fontsize=9.5, color=INK,
    )


def main():
    fold_factor = 4
    eeg, folded, label = load_tuev_example(fold_factor)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#98A2B3",
    })
    fig = plt.figure(figsize=(15.5, 12.0), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=(1.18, 0.98), hspace=0.55)
    top_grid = outer[0].subgridspec(
        1, 3, width_ratios=(1.45, 0.72, 1.28), wspace=0.26,
    )
    bottom_grid = outer[1].subgridspec(
        1, 3, width_ratios=(1.22, 0.28, 1.25), wspace=0.08,
    )

    raw_ax, map_ax, fold_ax = draw_concept_panels(fig, top_grid, fold_factor)
    draw_tuev_panel(fig, bottom_grid, eeg, folded, label, fold_factor)

    fig.suptitle(
        "Phase-Interleaved Lossless Temporal Folding", y=0.992,
        fontsize=19, weight="bold", color=INK,
    )
    fig.text(
        0.5, 0.505, "Same samples  ·  New adjacency  ·  Zero parameters",
        ha="center", fontsize=11.8, weight="bold", color="#155462",
    )
    fig.text(
        0.5, 0.470,
        "D. Concrete TUEV example: exact pipeline tensor before and after lossless P=4 folding",
        ha="center", fontsize=12.8, weight="bold", color=INK,
    )
    fig.subplots_adjust(top=0.925, bottom=0.07, left=0.058, right=0.975)

    for left, right in ((raw_ax, map_ax), (map_ax, fold_ax)):
        start = left.get_position().x1 + 0.006
        end = right.get_position().x0 - 0.006
        y = 0.5 * (left.get_position().y0 + left.get_position().y1)
        fig.add_artist(
            FancyArrowPatch(
                (start, y), (end, y), transform=fig.transFigure,
                arrowstyle="-|>", mutation_scale=18,
                linewidth=2.4, color=TEAL,
            )
        )

    output_dir = Path(__file__).resolve().parent
    fig.savefig(
        output_dir / "lossless_temporal_folding.png", dpi=300,
        facecolor="white", bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "lossless_temporal_folding.pdf",
        facecolor="white", bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
