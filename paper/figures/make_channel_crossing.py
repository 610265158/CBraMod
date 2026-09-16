"""A 3x3 window crossing an electrode block boundary after phase folding.

Panel (a) shows the folded layout with the window on the boundary; panel (b)
shows the same window's taps on the original traces, where rows obey
Delta t = P * Delta w + Delta p.  Real TUEV segment, P = 4.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from make_lossless_temporal_folding import load_tuev_example, TUEV_CHANNELS

PHASE = ["#2166AC", "#D97706", "#138A72", "#9B4C96"]
INK = "#1D2939"
MUTED = "#667085"
GRID = "#D9E0E8"
ORANGE = "#E68632"
MAGENTA = "#B25087"

FOLD = 4
TIME_LO, TIME_HI = 16, 48
WINDOW_COL = 6
FIRST_COL, N_COLS = 2, 14


def pick_channel_pair(eeg, lo, hi):
    scores = [
        (np.abs(eeg[c, lo:hi]).mean() + np.abs(eeg[c + 1, lo:hi]).mean(), c)
        for c in range(eeg.shape[0] - 1)
    ]
    return max(scores)[1]


def window_taps(window_col, fold):
    columns = (window_col - 1, window_col, window_col + 1)
    return {
        "c_p2": [fold * w + 2 for w in columns],
        "c_p3": [fold * w + 3 for w in columns],
        "n_p0": [fold * w for w in columns],
    }


def draw_folded_panel(ax, folded, ch, taps, window_col, first_col, n_cols):
    rows = folded[ch * FOLD:(ch + 2) * FOLD, first_col:first_col + n_cols]
    vmax = float(np.percentile(np.abs(rows), 99)) or 1.0
    ax.imshow(rows, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
              interpolation="nearest")
    for k in range(1, 2 * FOLD):
        ax.axhline(k - .5, color="white", lw=.7, alpha=.85)
    for x in np.arange(-.5, n_cols, 1):
        ax.axvline(x, color="white", lw=.6, alpha=.7)
    ax.axhline(FOLD - .5, color=INK, lw=2.3)
    ax.text(n_cols - .35, FOLD - .62, "block boundary", ha="right", va="bottom",
            fontsize=10, color=INK, fontweight="bold")

    wx = window_col - first_col
    ax.add_patch(Rectangle((wx - 1.5, FOLD - 2.5), 3, 3, fill=False,
                           ec=ORANGE, lw=2.6, zorder=6))
    ax.add_patch(Rectangle((wx - 1.5, FOLD - 3.5), 3, 3, fill=False,
                           ec=MAGENTA, lw=1.7, ls="--", zorder=6))
    ax.text(wx - 1.7, FOLD - 3.75, "in-block window (for contrast)",
            ha="left", va="bottom", fontsize=9, color=MAGENTA)
    for row, colour in [(FOLD - 2, PHASE[2]), (FOLD - 1, PHASE[3]),
                        (FOLD, PHASE[0])]:
        for w in (window_col - 1, window_col, window_col + 1):
            ax.plot(w - first_col, row, marker="o", ms=6.5, mfc=colour,
                    mec="white", mew=1.0, zorder=7)
    ax.annotate("this row reads\n" + TUEV_CHANNELS[ch + 1],
                xy=(wx + 1.5, FOLD), xytext=(wx + 2.1, FOLD + 2.5),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3),
                color=ORANGE, fontsize=9.6, fontweight="bold", ha="left")

    labels = [f"{TUEV_CHANNELS[ch]}, $p={p}$" for p in range(FOLD)] + \
             [f"{TUEV_CHANNELS[ch + 1]}, $p={p}$" for p in range(FOLD)]
    ax.set_yticks(range(2 * FOLD), labels)
    for tick, colour in zip(ax.get_yticklabels(), PHASE + PHASE):
        tick.set_color(colour)
    ax.set_xticks([0, 4, 8, 12],
                  [first_col, first_col + 4, first_col + 8, first_col + 12])
    ax.set_xlabel("folded column $w$")
    ax.set_title("(a) Folded layout at the block boundary",
                 loc="left", fontweight="bold")


def trace_for(ax, eeg, channel, lo, hi, offset):
    segment = eeg[channel, lo:hi]
    scale = max(float(np.abs(segment).max()), 1e-6)
    ax.plot(np.arange(lo, hi), segment / scale * .42 + offset,
            color="#4B5563", lw=1.0, zorder=2)
    ax.text(lo - 0.6, offset, TUEV_CHANNELS[channel], ha="right", va="center",
            fontsize=10.5, color=INK, fontweight="bold")
    return segment, scale


def tap_values(eeg, channel, taps, lo, hi, offset):
    segment = eeg[channel, lo:hi]
    scale = max(float(np.abs(segment).max()), 1e-6)
    return segment[[t - lo for t in taps]] / scale * .42 + offset


def draw_time_panel(ax, eeg, ch, taps, window_col, lo, hi):
    offsets = (1.15, -0.15)
    trace_for(ax, eeg, ch, lo, hi, offsets[0])
    trace_for(ax, eeg, ch + 1, lo, hi, offsets[1])

    ax.scatter(taps["c_p2"], tap_values(eeg, ch, taps["c_p2"], lo, hi, offsets[0]),
               s=34, color=PHASE[2], edgecolor="white", linewidth=.9, zorder=5)
    ax.scatter(taps["c_p3"], tap_values(eeg, ch, taps["c_p3"], lo, hi, offsets[0]),
               s=34, color=PHASE[3], edgecolor="white", linewidth=.9, zorder=5)
    ax.scatter(taps["n_p0"], tap_values(eeg, ch + 1, taps["n_p0"], lo, hi, offsets[1]),
               s=34, color=PHASE[0], edgecolor="white", linewidth=.9, zorder=5)

    span = taps["c_p3"]
    yy = tap_values(eeg, ch, span, lo, hi, offsets[0])
    ax.annotate("", xy=(span[-1], yy[-1]), xytext=(span[0], yy[0]),
                arrowprops=dict(arrowstyle="<->", color=PHASE[3], lw=1.2,
                                shrinkA=3, shrinkB=3))
    ax.text((span[0] + span[-1]) / 2, max(yy) + .22, r"$P=4$", ha="center",
            fontsize=10, color=PHASE[3], fontweight="bold")

    span = taps["n_p0"]
    yy = tap_values(eeg, ch + 1, span, lo, hi, offsets[1])
    ax.annotate("", xy=(span[-1], yy[-1]), xytext=(span[0], yy[0]),
                arrowprops=dict(arrowstyle="<->", color=PHASE[0], lw=1.2,
                                shrinkA=3, shrinkB=3))
    ax.text((span[0] + span[-1]) / 2, yy.mean() - .34, r"$P=4$", ha="center",
            fontsize=10, color=PHASE[0], fontweight="bold")

    center_t = FOLD * window_col + 3
    center_y = tap_values(eeg, ch, [center_t], lo, hi, offsets[0])[0]
    ax.scatter([center_t], [center_y], s=70, facecolor="none", edgecolor=INK,
               linewidth=1.2, zorder=6)
    ax.annotate("adjacent samples\n($\\Delta p=-1$)",
                xy=(center_t - .6, center_y - .06), xytext=(20.4, .60),
                fontsize=9.2, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    boundary_y = tap_values(eeg, ch + 1, [taps["n_p0"][1]], lo, hi, offsets[1])[0]
    ax.annotate("row $p=0$ of " + TUEV_CHANNELS[ch + 1] + "\n(block boundary)",
                xy=(taps["n_p0"][1], boundary_y),
                xytext=(20.8, -1.12), fontsize=9.2, color=ORANGE,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    ax.set_xlim(16.8, 35.5)
    ax.set_ylim(-1.30, 1.72)
    ax.set_xticks([18, 22, 26, 30, 34])
    ax.set_yticks([])
    ax.set_xlabel("original sample index $t$")
    ax.grid(axis="x", color=GRID, lw=.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#AEB8C5")
    ax.tick_params(colors=MUTED)
    ax.set_title("(b) Original-time taps of the window", loc="left",
                 fontweight="bold")
    handles = [
        Line2D([], [], ls="", marker="o", ms=6, mfc=PHASE[3], mec="white",
               label=r"$c,\ p=3$"),
        Line2D([], [], ls="", marker="o", ms=6, mfc=PHASE[2], mec="white",
               label=r"$c,\ p=2$"),
        Line2D([], [], ls="", marker="o", ms=6, mfc=PHASE[0], mec="white",
               label=r"$c+1,\ p=0$"),
    ]
    ax.text(center_t + .35, center_y + .12, "centre", fontsize=9, color=INK,
            fontweight="bold")
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9,
              ncol=1, handletextpad=.3, labelspacing=.35)


def main():
    eeg, folded, _ = load_tuev_example(FOLD)
    ch = pick_channel_pair(eeg, TIME_LO, TIME_HI)
    taps = window_taps(WINDOW_COL, FOLD)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 11.5,
        "axes.titlepad": 16, "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig = plt.figure(figsize=(9.2, 4.35), facecolor="white")
    grid = fig.add_gridspec(1, 2, left=.10, right=.99, bottom=.22, top=.80,
                            wspace=.28, width_ratios=[1.02, 1.0])
    folded_ax, time_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    draw_folded_panel(folded_ax, folded, ch, taps, WINDOW_COL, FIRST_COL, N_COLS)
    draw_time_panel(time_ax, eeg, ch, taps, WINDOW_COL, TIME_LO, TIME_HI)

    target = Path(__file__).resolve().parent / "channel_crossing"
    fig.savefig(target.with_suffix(".pdf"))
    fig.savefig(target.with_suffix(".png"), dpi=220)
    plt.close(fig)
    print(f"channels={TUEV_CHANNELS[ch]} / {TUEV_CHANNELS[ch+1]}")
    print(f"taps c={sorted(taps['c_p2'] + taps['c_p3'])} c+1={taps['n_p0']}")
    print(f"saved {target}")


if __name__ == "__main__":
    main()
