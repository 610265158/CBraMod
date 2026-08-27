"""Draw the core mechanism figure for phase-folded EEG and 2D CNNs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#172033"
MUTED = "#667085"
GRID = "#D9E0E8"
LIGHT = "#F5F8FB"
NAVY = "#285EA8"
TEAL = "#159A9C"
ORANGE = "#E68632"
MAGENTA = "#B25087"
GREEN = "#3B9C69"
PHASE = (NAVY, ORANGE, TEAL, MAGENTA)


def panel_label(ax, letter, title):
    ax.text(
        0.0, 1.055, letter, transform=ax.transAxes, fontsize=14.5,
        weight="bold", color="white", va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=INK, edgecolor="none"),
    )
    ax.text(
        0.105, 1.055, title, transform=ax.transAxes, fontsize=15,
        weight="bold", color=INK, va="center", ha="left",
    )


def clean_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AEB8C5")
    ax.tick_params(colors=MUTED, labelsize=8)


def draw_raw(ax, fold_factor=4):
    panel_label(ax, "A", "Raw EEG is one-dimensional")
    rng = np.random.default_rng(7)
    t = np.arange(40)
    slow = 0.22 * np.sin(2 * np.pi * t / 14)
    spike = 1.15 * np.exp(-0.5 * ((t - 19.0) / 0.72) ** 2)
    after = -0.48 * np.exp(-0.5 * ((t - 22.0) / 2.0) ** 2)
    x0 = slow + spike + after + 0.025 * rng.normal(size=t.size)
    x1 = 0.18 * np.sin(2 * np.pi * (t - 2) / 14) + 0.54 * spike + 0.02 * rng.normal(size=t.size)
    offsets = (1.05, -0.25)

    ax.plot(t, x0 + offsets[0], color=INK, lw=1.65, zorder=2)
    ax.plot(t, x1 + offsets[1], color=INK, lw=1.35, alpha=0.88, zorder=2)
    for p in range(fold_factor):
        keep = t % fold_factor == p
        ax.scatter(
            t[keep], x0[keep] + offsets[0], s=24, color=PHASE[p],
            edgecolor="white", linewidth=0.65, zorder=3,
        )

    ax.axvspan(15.6, 24.4, color=ORANGE, alpha=0.09, lw=0)
    ax.text(20, 2.35, "local morphology", ha="center", color=ORANGE,
            fontsize=9.5, weight="bold")
    ax.annotate(
        "spike + slow wave", xy=(19, x0[19] + offsets[0]), xytext=(9, 2.05),
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
        color=ORANGE, fontsize=9,
    )
    ax.text(-2.2, 1.05, r"$c$", ha="right", va="center", color=NAVY,
            fontsize=10.5, weight="bold")
    ax.text(-2.2, -0.25, r"$c+1$", ha="right", va="center", color=NAVY,
            fontsize=10.5, weight="bold")
    ax.set_xlim(-0.7, 39.7)
    ax.set_ylim(-0.9, 2.65)
    ax.set_xlabel("sample index  $t$", color=INK, fontsize=9.5)
    ax.set_ylabel("electrode / amplitude", color=INK, fontsize=9.5)
    ax.set_xticks([0, 8, 16, 24, 32, 39])
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=0.55)
    clean_axis(ax)

    y = -0.72
    for p, color in enumerate(PHASE):
        ax.scatter(4 + p * 8.0, y, s=31, color=color, clip_on=False)
        ax.text(5.1 + p * 8.0, y, rf"phase $p={p}$", va="center",
                fontsize=8.2, color=MUTED, clip_on=False)
    ax.text(
        0.5, -0.22, r"Input geometry: $X\in\mathbb{R}^{C\times T}$",
        transform=ax.transAxes, ha="center", fontsize=10.5, color=INK,
        weight="bold",
    )


def draw_folded(ax, fold_factor=4):
    panel_label(ax, "B", r"Lossless phase folding  ($P=4$)")
    rows, cols = 8, 10
    xx = np.linspace(0, 2.4 * np.pi, cols)
    values = np.vstack([
        np.sin(xx + 0.42 * r) + 0.28 * np.cos(2.1 * xx - 0.25 * r)
        for r in range(rows)
    ])
    ax.imshow(values, cmap="RdBu_r", aspect="auto", interpolation="nearest",
              vmin=-1.35, vmax=1.35, origin="upper")

    for x in np.arange(-0.5, cols, 1):
        ax.axvline(x, color="white", lw=0.7, alpha=0.75)
    for y in np.arange(-0.5, rows, 1):
        ax.axhline(y, color="white", lw=0.7, alpha=0.75)
    ax.axhline(3.5, color=INK, lw=2.1)

    labels = [
        rf"$c,\ p={p}$" for p in range(fold_factor)
    ] + [
        rf"$c+1,\ p={p}$" for p in range(fold_factor)
    ]
    ax.set_yticks(range(rows), labels, fontsize=8.5)
    for tick, color in zip(ax.get_yticklabels(), PHASE + PHASE):
        tick.set_color(color)
        tick.set_weight("bold")
    ax.set_xticks([0, 2, 4, 6, 8, 9], ["0", "2", "4", "6", "8", "9"])
    ax.set_xlabel(r"folded-time column  $w$", fontsize=9.5, color=INK)
    ax.tick_params(length=0)

    # A same-electrode 3x3 window: rows p=0,1,2 around p=1.
    ax.add_patch(Rectangle(
        (2.5, -0.5), 3, 3, fill=False, edgecolor=ORANGE,
        linewidth=3.0, zorder=5,
    ))
    ax.text(5.7, 0.95, "same-electrode\n3×3 window", color=ORANGE,
            fontsize=8.8, weight="bold", va="center")

    # A boundary window exposes the second, cross-electrode adjacency.
    ax.add_patch(Rectangle(
        (6.5, 2.5), 3, 3, fill=False, edgecolor=MAGENTA,
        linewidth=2.5, linestyle="--", zorder=5,
    ))
    ax.text(6.45, 5.85, "boundary window →\nelectrode mixing", color=MAGENTA,
            fontsize=8.5, weight="bold", ha="left")

    ax.text(
        0.5, -0.19, r"$I[cP+p,w]=X[c,wP+p]$",
        transform=ax.transAxes, ha="center", fontsize=12.2, color=INK,
        weight="bold",
    )
    ax.text(
        0.5, -0.285, r"$[C,T]\;\longrightarrow\;[CP,T/P]$  ·  permutation only",
        transform=ax.transAxes, ha="center", fontsize=9.7, color=TEAL,
    )


def draw_offset_grid(ax):
    panel_label(ax, "C", "A 2D kernel becomes multi-lag temporal filtering")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Rows move vertically through phase (delta p); columns move horizontally
    # through folded time (delta w).  Hence delta t = P*delta w + delta p.
    offsets = np.array([[-5, -1, 3], [-4, 0, 4], [-3, 1, 5]])
    x0, y0, size = 0.65, 4.55, 1.42
    row_names = [r"$\Delta p=-1$", r"$\Delta p=0$", r"$\Delta p=+1$"]
    col_names = [r"$\Delta w=-1$", r"$\Delta w=0$", r"$\Delta w=+1$"]
    for row in range(3):
        for col in range(3):
            x = x0 + col * size
            y = y0 + (2 - row) * size
            is_center = row == 1 and col == 1
            face = "#FFF2E8" if is_center else "#EDF6F6"
            edge = ORANGE if is_center else TEAL
            ax.add_patch(Rectangle((x, y), size, size, facecolor=face,
                                   edgecolor=edge, lw=1.6))
            value = offsets[row, col]
            shown = "0" if value == 0 else f"{value:+d}"
            ax.text(x + size / 2, y + size / 2 + 0.12, shown,
                    ha="center", va="center", fontsize=15, color=INK,
                    weight="bold")
            ax.text(x + size / 2, y + 0.22, r"samples",
                    ha="center", va="bottom", fontsize=7.0, color=MUTED)
    for row, name in enumerate(row_names):
        y = y0 + (2 - row) * size + size / 2
        ax.text(x0 - 0.18, y, name, ha="right", va="center",
                fontsize=8.5, color=PHASE[min(row, 3)])
    for col, name in enumerate(col_names):
        x = x0 + col * size + size / 2
        ax.text(x, y0 + 3 * size + 0.22, name, ha="center", va="bottom",
                fontsize=8.3, color=MUTED)

    ax.text(x0 + 1.5 * size, 9.48, "original-time offsets",
            ha="center", fontsize=10.2, weight="bold", color=INK)
    ax.text(
        7.0, 8.75, r"$\Delta t=P\Delta w+\Delta p$",
        ha="center", fontsize=17, color=INK, weight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="#F7FAFC", ec=GRID, lw=1.2),
    )
    ax.text(7.0, 7.73, r"for $P=4$", ha="center", fontsize=10, color=MUTED)

    cards = [
        (TEAL, "vertical", "adjacent phases", r"$\Delta t\approx\pm1$"),
        (NAVY, "horizontal", "$P$-step context", r"$\Delta t=\pm4$"),
        (ORANGE, "diagonal", "mixed offsets", r"$\Delta t=\pm3,\pm5$"),
    ]
    for i, (color, title, body, math) in enumerate(cards):
        bx = 5.15 + (i % 2) * 3.35
        by = 4.60 - (i // 2) * 2.28
        if i == 2:
            bx = 6.83
        ax.add_patch(FancyBboxPatch(
            (bx, by), 3.0, 1.65, boxstyle="round,pad=0.16",
            facecolor="white", edgecolor=color, lw=1.45,
        ))
        ax.add_patch(Rectangle((bx, by), 0.13, 1.65, facecolor=color,
                               edgecolor="none"))
        ax.text(bx + 0.28, by + 1.20, title, color=color, fontsize=9.5,
                weight="bold", va="center")
        ax.text(bx + 0.28, by + 0.78, body, color=INK, fontsize=8.6,
                va="center")
        ax.text(bx + 0.28, by + 0.30, math, color=MUTED, fontsize=9.0,
                va="center")

    ax.text(
        6.7, 0.72,
        "One local square sees slopes, phases, and dilated context at once.",
        ha="center", fontsize=10.2, color=INK, weight="bold",
    )


def rounded_card(ax, xy, width, height, color, title, scale, function):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.025,rounding_size=0.025",
        transform=ax.transAxes, facecolor="white", edgecolor="#CDD6E1", lw=1.1,
    ))
    ax.add_patch(Rectangle(
        (x, y + height - 0.042), width, 0.042, transform=ax.transAxes,
        facecolor=color, edgecolor="none",
    ))
    ax.text(x + 0.026, y + height - 0.11, title, transform=ax.transAxes,
            fontsize=10.0, color=color, weight="bold", va="top")
    ax.text(x + 0.026, y + height - 0.235, scale, transform=ax.transAxes,
            fontsize=9.1, color=INK, weight="bold", va="top")
    ax.text(x + 0.026, y + 0.08, function, transform=ax.transAxes,
            fontsize=8.6, color=MUTED, va="bottom", linespacing=1.28)


def draw_hierarchy(ax):
    panel_label(ax, "D", "EfficientNet-B0 grows from waveform morphology to global context")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # The small raster at the left makes the growing 2D receptive field explicit.
    gx, gy, gw, gh = 0.015, 0.16, 0.16, 0.62
    mini = np.sin(np.linspace(0, 8 * np.pi, 24))[None, :] * np.linspace(0.5, 1.0, 12)[:, None]
    ax.imshow(mini, extent=(gx, gx + gw, gy, gy + gh), transform=ax.transAxes,
              cmap="RdBu_r", aspect="auto", interpolation="nearest",
              vmin=-1, vmax=1, zorder=0)
    for frac, color, lw in [(0.18, ORANGE, 2.0), (0.42, TEAL, 1.8), (0.78, NAVY, 1.8)]:
        width = gw * frac
        height = gh * frac
        ax.add_patch(Rectangle(
            (gx + 0.5 * (gw - width), gy + 0.5 * (gh - height)), width, height,
            transform=ax.transAxes, fill=False, ec=color, lw=lw, zorder=2,
        ))
    ax.text(gx + gw / 2, 0.085, "folded EEG", transform=ax.transAxes,
            ha="center", fontsize=8.5, color=MUTED)

    starts = [0.205, 0.385, 0.565, 0.745]
    widths = [0.155, 0.155, 0.155, 0.235]
    rounded_card(ax, (starts[0], 0.17), widths[0], 0.60, ORANGE,
                 "Stem / early blocks", "RF 3–7 px  →  55–140 ms",
                 "spikes · edges\nslopes · local phase")
    rounded_card(ax, (starts[1], 0.17), widths[1], 0.60, TEAL,
                 "Middle blocks", "RF 19 px  →  380 ms",
                 "waveform complexes\nshort oscillations")
    rounded_card(ax, (starts[2], 0.17), widths[2], 0.60, NAVY,
                 "Deep blocks", "RF 67–147 px  →  1.34–2.94 s",
                 "bursts · rhythms\ncross-electrode context")
    rounded_card(ax, (starts[3], 0.17), widths[3], 0.60, MAGENTA,
                 "Final representation", "RF 851 px  →  17.0 s / token",
                 "token aggregation → whole segment\n→ task decision")

    for left, right in zip(starts[:-1], starts[1:]):
        end = left + widths[starts.index(left)]
        ax.add_patch(FancyArrowPatch(
            (end + 0.005, 0.47), (right - 0.008, 0.47),
            transform=ax.transAxes, arrowstyle="-|>", mutation_scale=11,
            lw=1.25, color="#8B98A9",
        ))

    ax.text(
        0.59, 0.91,
        r"Temporal span:  $R_t(P)=P(R_w-1)+\min(R_h,P)$   (within one electrode)",
        transform=ax.transAxes, ha="center", fontsize=10.2, color=INK,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=LIGHT, edgecolor=GRID),
    )
    ax.text(
        0.59, 0.825,
        r"Example scales use $P=4$ and 200 Hz; vertical RF growth also crosses electrode groups.",
        transform=ax.transAxes, ha="center", fontsize=8.8, color=MUTED,
    )


def add_flow_arrow(fig, left_ax, right_ax, label):
    a = left_ax.get_position()
    b = right_ax.get_position()
    y = (a.y0 + a.y1) / 2
    start = (a.x1 + 0.008, y)
    end = (b.x0 - 0.008, y)
    fig.add_artist(FancyArrowPatch(
        start, end, transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=18, lw=2.2, color="#8190A4",
    ))
    fig.text((start[0] + end[0]) / 2, y + 0.033, label, ha="center",
             va="bottom", fontsize=8.5, color=MUTED, weight="bold")


def main():
    fig = plt.figure(figsize=(17.2, 9.6), facecolor="white")
    grid = fig.add_gridspec(
        2, 3, height_ratios=(1.12, 0.88), width_ratios=(1.05, 1.12, 1.10),
        left=0.048, right=0.978, bottom=0.095, top=0.865,
        hspace=0.43, wspace=0.25,
    )
    raw_ax = fig.add_subplot(grid[0, 0])
    fold_ax = fig.add_subplot(grid[0, 1])
    offset_ax = fig.add_subplot(grid[0, 2])
    hierarchy_ax = fig.add_subplot(grid[1, :])

    draw_raw(raw_ax)
    draw_folded(fold_ax)
    draw_offset_grid(offset_ax)
    draw_hierarchy(hierarchy_ax)
    add_flow_arrow(fig, raw_ax, fold_ax, "reindex")
    add_flow_arrow(fig, fold_ax, offset_ax, "convolve")

    fig.suptitle(
        "Why Phase-Folded EEG Fits the Receptive Fields of a 2D CNN",
        x=0.5, y=0.965, fontsize=22, weight="bold", color=INK,
    )
    fig.text(
        0.5, 0.916,
        "A lossless permutation exposes polyphase temporal structure as local 2D neighborhoods, "
        "then CNN depth integrates progressively longer and wider EEG context.",
        ha="center", fontsize=11.2, color=MUTED,
    )
    fig.text(
        0.5, 0.028,
        "Core insight: folding changes adjacency—not information—so pretrained visual filters receive an EEG geometry they can exploit.",
        ha="center", fontsize=12.0, color="white", weight="bold",
        bbox=dict(boxstyle="round,pad=0.55", facecolor=INK, edgecolor="none"),
    )

    output = Path(__file__).resolve().parent
    stem = output / "receptive_field_mechanism"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white",
                bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white",
                bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white",
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
