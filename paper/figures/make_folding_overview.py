"""Compose the merged folding + receptive-field overview figure.

Produces paper/figures/folding_overview.{pdf,png} at final print size
(7.0 x 3.6 in) so the fonts stay readable when LaTeX includes it at text width.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_lossless_temporal_folding import load_tuev_example

INK = "#172033"
MUTED = "#667085"
NAVY = "#285EA8"
TEAL = "#159A9C"
ORANGE = "#E68632"
MAGENTA = "#B25087"
PHASE = (NAVY, ORANGE, TEAL, MAGENTA)


def panel_title(ax, letter, text, pad=4):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=8.6,
                 weight="bold", color=INK, pad=pad)


def panel_raw(ax):
    rng = np.random.default_rng(7)
    t = np.arange(40)
    slow = 0.22 * np.sin(2 * np.pi * t / 14)
    spike = 1.15 * np.exp(-0.5 * ((t - 19.0) / 0.72) ** 2)
    after = -0.48 * np.exp(-0.5 * ((t - 22.0) / 2.0) ** 2)
    y = slow + spike + after + 0.02 * rng.normal(size=t.size)
    ax.plot(t, y, color=INK, lw=1.0, zorder=2)
    for phase in range(4):
        keep = t % 4 == phase
        ax.scatter(t[keep], y[keep], s=9, color=PHASE[phase],
                   edgecolor="white", linewidth=0.3, zorder=3)
    ax.axvspan(15.6, 24.4, color=ORANGE, alpha=0.10, lw=0)
    ax.annotate("spike + slow wave", xy=(19, y[19]), xytext=(3.2, 1.42),
                fontsize=6.6, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
    ax.text(0.985, 0.03, r"colors: $p = t\ \mathrm{mod}\ 4$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.3, color=MUTED)
    ax.set_xlim(-0.6, 39.6)
    ax.set_ylim(-1.05, 1.72)
    ax.set_xlabel("sample index $t$", fontsize=7.2)
    ax.set_ylabel("amplitude", fontsize=7.2)
    ax.set_yticks([])
    ax.tick_params(labelsize=6.3, length=2, pad=1)
    ax.grid(axis="x", color="#E4E9F0", lw=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    panel_title(ax, "a", "Raw EEG is one-dimensional")


def panel_real_fold(ax, fold_factor=4):
    eeg, folded, _ = load_tuev_example(fold_factor)
    limit = float(np.percentile(np.abs(folded), 99.5))
    ax.imshow(np.clip(folded, -limit, limit), aspect="auto",
              interpolation="nearest", cmap="RdBu_r",
              vmin=-limit, vmax=limit, origin="upper")
    for boundary in range(fold_factor, folded.shape[0], fold_factor):
        ax.axhline(boundary - 0.5, color="white", linewidth=0.5, alpha=0.9)
    ticks = list(range(1, folded.shape[0], 8))
    ax.set_yticks(ticks, [f"ch{i // 4 + 1}" for i in ticks], fontsize=5.6)
    for tick, row in zip(ax.get_yticklabels(), ticks):
        tick.set_color(PHASE[row % fold_factor])
    ax.set_xticks([0, 50, 100, 150, 200, 249],
                  ["0", "1", "2", "3", "4", "5"])
    ax.tick_params(labelsize=6.0, length=2, pad=1)
    ax.set_xlabel("folded time (s)", fontsize=7.2)
    ax.set_ylabel("16 electrodes $\\times$ 4 phases", fontsize=6.8)
    ax.text(0.99, 0.02, r"$I[cP{+}p,w]=X[c,wP{+}p]$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.2, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2))
    panel_title(ax, "b", "Real TUEV: lossless fold ($P=4$)")


def panel_offsets(ax):
    ax.set_axis_off()
    ax.set_xlim(-1.7, 6.1)
    ax.set_ylim(-1.35, 4.35)
    values = np.array([[-5, -1, 3], [-4, 0, 4], [-3, 1, 5]])
    size = 1.12
    for row in range(3):
        for col in range(3):
            x = col * size
            y = (2 - row) * size
            center = row == 1 and col == 1
            ax.add_patch(Rectangle(
                (x, y), size - 0.12, size - 0.12,
                facecolor="#FFF2E8" if center else "#EDF6F6",
                edgecolor=ORANGE if center else TEAL, lw=1.0))
            ax.text(x + (size - 0.12) / 2, y + (size - 0.12) / 2,
                    f"{values[row, col]:+d}", ha="center", va="center",
                    fontsize=8.4, weight="bold", color=INK)
    ax.text(-1.35, 1.5, r"$\Delta p$", rotation=90, va="center",
            fontsize=7.4, color=MUTED)
    ax.text(1.5, 3.55, r"$\Delta w$", ha="center", fontsize=7.4, color=MUTED)
    ax.text(1.5, -0.72, r"$\Delta t = P\,\Delta w + \Delta p$",
            ha="center", fontsize=7.8, color=INK, weight="bold")
    ax.text(1.5, -1.18, "one $3\\times3$ patch: $\\pm1$-sample phases\nand $\\pm4$-sample context",
            ha="center", va="top", fontsize=6.4, color=MUTED)
    panel_title(ax, "c", "Multi-lag $3\\times3$ filtering", pad=2)


def panel_hierarchy(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cards = [
        (ORANGE, "Stem / early", "3–7 px  →  55–140 ms",
         "spikes, slopes,\nlocal phase"),
        (TEAL, "Middle", "19 px  →  380 ms",
         "waveform complexes,\nshort oscillations"),
        (NAVY, "Deep", "67–147 px  →  1.3–2.9 s",
         "bursts, rhythms,\ncross-electrode"),
        (MAGENTA, "Aggregation", "851 px  →  17 s / token",
         "whole segment\n→ task decision"),
    ]
    width, gap = 0.235, 0.02
    starts = [i * (width + gap) for i in range(4)]
    for start, (color, title, scale, function) in zip(starts, cards):
        ax.add_patch(FancyBboxPatch(
            (start, 0.12), width, 0.58,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            transform=ax.transAxes, facecolor="white", edgecolor="#CDD6E1",
            lw=0.8))
        ax.add_patch(Rectangle(
            (start, 0.64), width, 0.055, transform=ax.transAxes,
            facecolor=color, edgecolor="none"))
        ax.text(start + 0.012, 0.565, title, transform=ax.transAxes,
                fontsize=7.4, weight="bold", color=color, va="top")
        ax.text(start + 0.012, 0.435, scale, transform=ax.transAxes,
                fontsize=6.6, weight="bold", color=INK, va="top")
        ax.text(start + 0.012, 0.325, function, transform=ax.transAxes,
                fontsize=6.2, color=MUTED, va="top", linespacing=1.25)
    for start in starts[:-1]:
        ax.add_patch(FancyArrowPatch(
            (start + width + 0.002, 0.42), (start + width + gap - 0.002, 0.42),
            transform=ax.transAxes, arrowstyle="-|>", mutation_scale=7,
            lw=0.9, color="#8B98A9"))
    panel_title(ax, "d", "EfficientNet-B0: receptive fields grow with depth",
                pad=1)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.edgecolor": "#98A2B3",
    })
    fig = plt.figure(figsize=(7.0, 3.6), facecolor="white")
    grid = fig.add_gridspec(
        2, 3, height_ratios=(1.0, 0.78), width_ratios=(1.0, 1.22, 1.0),
        left=0.052, right=0.99, bottom=0.085, top=0.925,
        hspace=0.62, wspace=0.30,
    )
    raw_ax = fig.add_subplot(grid[0, 0])
    fold_ax = fig.add_subplot(grid[0, 1])
    offset_ax = fig.add_subplot(grid[0, 2])
    hierarchy_ax = fig.add_subplot(grid[1, :])

    panel_raw(raw_ax)
    panel_real_fold(fold_ax)
    panel_offsets(offset_ax)
    panel_hierarchy(hierarchy_ax)

    output = Path(__file__).resolve().parent / "folding_overview"
    fig.savefig(str(output) + ".png", dpi=300, facecolor="white",
                bbox_inches="tight")
    fig.savefig(str(output) + ".pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
