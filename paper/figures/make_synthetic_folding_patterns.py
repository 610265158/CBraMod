"""Generate controlled synthetic EEG morphologies before/after P=4 folding."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


INK = "#1D2939"
MUTED = "#667085"
PHASE_COLORS = ("#2E77B5", "#D8752B", "#239B75", "#8558B5")
FS = 200
SECONDS = 5
P = 4


def gaussian(t, center, width):
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def make_patterns():
    rng = np.random.default_rng(19)
    t = np.arange(FS * SECONDS) / FS

    # Repeated sharp transient followed by a broad slow wave.
    spike_slow = 0.025 * rng.standard_normal(t.size)
    for center in np.arange(0.55, 4.75, 0.70):
        spike_slow += 1.15 * gaussian(t, center, 0.010)
        spike_slow -= 0.72 * gaussian(t, center + 0.105, 0.085)
    spike_slow += 0.04 * np.sin(2 * np.pi * 0.7 * t)

    # Rhythmic ictal activity with a smooth onset and offset envelope.
    envelope = 0.08 + 0.92 * np.exp(-0.5 * ((t - 2.55) / 1.25) ** 2)
    rhythmic = envelope * np.sin(2 * np.pi * 4.0 * t + 0.2)
    rhythmic += 0.035 * rng.standard_normal(t.size)

    # Periodic discharges: regular sharp waves with after-going slow activity.
    periodic = 0.025 * rng.standard_normal(t.size)
    for center in np.arange(0.45, 4.95, 0.90):
        periodic += 0.95 * gaussian(t, center, 0.016)
        periodic -= 0.42 * gaussian(t, center + 0.16, 0.11)
    periodic += 0.04 * np.sin(2 * np.pi * 0.35 * t)

    return t, [spike_slow, rhythmic, periodic]


def fold(signal):
    width = signal.size // P
    return signal.reshape(width, P).T


def draw_raw(ax, t, y, title, descriptor):
    ax.plot(t, y, color=INK, linewidth=0.75)
    ax.axhline(0, color="#D0D5DD", linewidth=0.6)
    ax.set_xlim(0, SECONDS)
    limit = max(np.percentile(np.abs(y), 99.5) * 1.12, 0.2)
    ax.set_ylim(-limit, limit)
    ax.set_title(title, loc="left", pad=4, fontsize=10.5, weight="bold")
    ax.text(
        0.01, 0.92, descriptor, transform=ax.transAxes,
        ha="left", va="top", fontsize=7.7, color=MUTED,
    )
    ax.grid(axis="x", color="#EAECF0", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def draw_heatmap(ax, folded):
    limit = max(np.percentile(np.abs(folded), 99.5), 1e-6)
    image = ax.imshow(
        np.clip(folded, -limit, limit), aspect="auto", interpolation="nearest",
        cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper",
    )
    for row in range(P):
        ax.add_patch(Rectangle(
            (-10, row - 0.5), 7, 1, facecolor=PHASE_COLORS[row],
            edgecolor="none", clip_on=False,
        ))
    ax.set_xlim(-0.5, folded.shape[1] - 0.5)
    ax.set_ylim(P - 0.5, -0.5)
    ax.set_xticks([0, 50, 100, 150, 200, 249], ["0", "1", "2", "3", "4", "5"])
    ax.set_yticks(range(P), [f"p={p}" for p in range(P)], fontsize=7)
    for tick, color in zip(ax.get_yticklabels(), PHASE_COLORS):
        tick.set_color(color)
        tick.set_weight("bold")
    ax.grid(axis="x", color="white", linewidth=0.45, alpha=0.7)
    ax.tick_params(length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    return image


def draw_phase_rows(ax, folded):
    offsets = np.arange(P - 1, -1, -1, dtype=float)
    x = np.arange(folded.shape[1]) / (FS / P)
    gain = 0.34 / max(np.percentile(np.abs(folded), 99.5), 1e-6)
    for phase in range(P):
        base = offsets[phase]
        ax.axhspan(base - 0.42, base + 0.42, color=PHASE_COLORS[phase], alpha=0.09)
        ax.axhline(base, color="#D0D5DD", linewidth=0.5)
        ax.plot(x, base + folded[phase] * gain, color=PHASE_COLORS[phase], linewidth=0.75)
    ax.set_xlim(0, SECONDS)
    ax.set_ylim(-0.48, P - 0.52)
    ax.set_yticks(offsets, [f"p={p}" for p in range(P)], fontsize=7)
    for tick, color in zip(ax.get_yticklabels(), PHASE_COLORS):
        tick.set_color(color)
        tick.set_weight("bold")
    ax.grid(axis="x", color="#EAECF0", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    t, patterns = make_patterns()
    titles = [
        "A. Spike–slow-wave complex",
        "B. Rhythmic ictal oscillation",
        "C. Periodic discharges",
    ]
    descriptors = [
        "narrow spike → broad after-going slow wave",
        "4 Hz rhythm with gradual recruitment and termination",
        "regular sharp transient with slow after-wave",
    ]
    fig = plt.figure(figsize=(15.5, 10.2), facecolor="white")
    grid = fig.add_gridspec(3, 3, width_ratios=(1.22, 0.92, 1.22),
                            hspace=0.48, wspace=0.22)
    for row, (y, title, descriptor) in enumerate(zip(patterns, titles, descriptors)):
        raw_ax = fig.add_subplot(grid[row, 0])
        heat_ax = fig.add_subplot(grid[row, 1])
        phase_ax = fig.add_subplot(grid[row, 2])
        folded = fold(y)
        draw_raw(raw_ax, t, y, title, descriptor)
        draw_heatmap(heat_ax, folded)
        draw_phase_rows(phase_ax, folded)
        if row == 2:
            raw_ax.set_xlabel("time (s), 200 Hz")
            heat_ax.set_xlabel("folded time (s), one column = 20 ms", fontsize=8)
            phase_ax.set_xlabel("folded time (s), horizontal step = P=4 samples", fontsize=8)
        else:
            raw_ax.set_xticklabels([])
            heat_ax.set_xticklabels([])
            phase_ax.set_xticklabels([])

    fig.text(0.44, 0.968, "Simulated EEG Morphologies under Lossless P=4 Folding",
             ha="center", fontsize=18, weight="bold", color=INK)
    fig.text(
        0.44, 0.936,
        "Each row retains exactly 1,000 samples; folding changes temporal adjacency, not morphology or information.",
        ha="center", fontsize=10.5, weight="bold", color="#155462",
    )
    fig.text(0.645, 0.905, "P=4 folded tensor", ha="center", fontsize=10, color=MUTED)
    fig.text(0.865, 0.905, "phase rows seen by the CNN", ha="center", fontsize=10, color=MUTED)
    fig.subplots_adjust(top=0.88, bottom=0.075, left=0.06, right=0.975)
    output_dir = Path(__file__).resolve().parent
    fig.savefig(output_dir / "synthetic_folding_patterns.png", dpi=300,
                facecolor="white", bbox_inches="tight")
    fig.savefig(output_dir / "synthetic_folding_patterns.pdf",
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
