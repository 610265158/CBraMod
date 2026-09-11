#!/usr/bin/env python
"""Build the strict analysis bundle for the ViT cls_token+lr5e-4+clip1 rerun.

Parses the completed run logs (12 datasets x 5 seeds), compares against the
previously recorded baselines (flatten@1e-4 formal / patch-mean("cls_token")@1e-4
head comparisons), runs Welch t-tests from summary statistics with Holm
correction, computes Cohen's d, and generates real figures.

Notes / validators:
- New runs: per-seed test values parsed from queue logs; sample std (ddof=1).
- Old baselines: YAML records population stds (n=5); converted to sample stds
  (x sqrt(n/(n-1))) before being used in Welch tests.
- Old per-seed values are unavailable -> only unpaired tests from summary
  statistics are possible. Stated as a limitation.
"""
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

LOG_BASE = Path("/tmp/codemaker/vit_rerun_2026-09-11")
OUT = Path("/data/lz/public/CBraMod/experiments/analysis/vit_clstoken_lr5e4_clip1_2026-09-11")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PRIMARY = {
    "CHB-MIT": "pr_auc", "TUAB": "pr_auc", "SHU-MI": "pr_auc",
    "Mumtaz2016": "pr_auc", "MentalArithmetic": "pr_auc",
    "TUEV": "kappa", "ISRUC": "kappa", "FACED": "kappa", "SEED-V": "kappa",
    "PhysioNet-MI": "kappa", "BCIC2020-3": "kappa", "HMC": "kappa",
}

# log file per dataset (batch1 prefix clstoken_, batch2 prefix rem_)
LOGS = {
    "CHB-MIT": "rem_CHB-MIT.queue.log", "TUAB": "clstoken_TUAB.queue.log",
    "SHU-MI": "clstoken_SHU-MI.queue.log", "Mumtaz2016": "clstoken_Mumtaz2016.queue.log",
    "MentalArithmetic": "rem_MentalArithmetic.queue.log", "TUEV": "clstoken_TUEV.queue.log",
    "ISRUC": "clstoken_ISRUC.queue.log", "FACED": "clstoken_FACED.queue.log",
    "SEED-V": "clstoken_SEED-V.queue.log", "PhysioNet-MI": "rem_PhysioNet-MI.queue.log",
    "BCIC2020-3": "rem_BCIC2020-3.queue.log", "HMC": "rem_HMC.queue.log",
}

# mean, pop-std (n=5) -- from the YAML configs (source of truth)
OLD_FLATTEN = {
    "CHB-MIT": {"ba": (0.72782, 0.01053), "pr_auc": (0.45657, 0.01308), "roc_auc": (0.90536, 0.01143)},
    "TUAB": {"ba": (0.80491, 0.00409), "pr_auc": (0.88428, 0.00397), "roc_auc": (0.87591, 0.00522)},
    "TUEV": {"ba": (0.72170, 0.00621), "kappa": (0.72625, 0.00426), "f1": (0.85629, 0.00293)},
    "ISRUC": {"ba": (0.80575, 0.00168), "kappa": (0.77854, 0.00277), "f1": (0.82864, 0.00207)},
    "FACED": {"ba": (0.38672, 0.00711), "kappa": (0.30969, 0.00717), "f1": (0.38489, 0.00711)},
    "SEED-V": {"ba": (0.39298, 0.00351), "kappa": (0.24419, 0.00450), "f1": (0.40163, 0.00378)},
    "PhysioNet-MI": {"ba": (0.57728, 0.00524), "kappa": (0.43634, 0.00698), "f1": (0.57740, 0.00604)},
    "SHU-MI": {"ba": (0.59568, 0.00959), "pr_auc": (0.66185, 0.01610), "roc_auc": (0.63474, 0.01334)},
    "BCIC2020-3": {"ba": (0.66187, 0.00903), "kappa": (0.57733, 0.01129), "f1": (0.66180, 0.00892)},
    "Mumtaz2016": {"ba": (0.90721, 0.01520), "pr_auc": (0.98067, 0.00161), "roc_auc": (0.97880, 0.00213)},
    "MentalArithmetic": {"ba": (0.71181, 0.03534), "pr_auc": (0.57394, 0.06172), "roc_auc": (0.76186, 0.03888)},
    "HMC": {"ba": (0.74744, 0.00321), "kappa": (0.69805, 0.00157), "f1": (0.76428, 0.00059)},
}

OLD_PATCHMEAN = {
    "CHB-MIT": {"ba": (0.73342, 0.01613), "pr_auc": (0.46393, 0.03429), "roc_auc": (0.91384, 0.00936)},
    "TUAB": {"ba": (0.82262, 0.00073), "pr_auc": (0.91141, 0.00188), "roc_auc": (0.90658, 0.00138)},
    "ISRUC": {"ba": (0.80896, 0.00317), "kappa": (0.78255, 0.00352), "f1": (0.83176, 0.00261)},
    "FACED": {"ba": (0.29197, 0.00627), "kappa": (0.20305, 0.00720), "f1": (0.29206, 0.00705)},
    "SEED-V": {"ba": (0.39551, 0.00453), "kappa": (0.24613, 0.00558), "f1": (0.40228, 0.00408)},
    "PhysioNet-MI": {"ba": (0.60335, 0.00981), "kappa": (0.47110, 0.01308), "f1": (0.60467, 0.01048)},
    "SHU-MI": {"ba": (0.60375, 0.00707), "pr_auc": (0.66889, 0.00318), "roc_auc": (0.65150, 0.01181)},
    "BCIC2020-3": {"ba": (0.60213, 0.01705), "kappa": (0.50267, 0.02131), "f1": (0.60205, 0.01716)},
    "MentalArithmetic": {"ba": (0.75486, 0.04011), "pr_auc": (0.69924, 0.03928), "roc_auc": (0.82188, 0.02365)},
}


def parse_log(path):
    cur, tests = None, {}
    for line in path.read_text(errors="replace").splitlines():
        m = re.search(r"finetune_main\.py.*--seed (\d+)", line)
        if m:
            cur = int(m.group(1))
        m = re.search(r"Test Evaluation:\s*(.+)", line)
        if m and cur is not None:
            tests[cur] = {k: float(v) for k, v in re.findall(r"(\w+):\s*([\d.]+)", m.group(1))}
    return tests


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def welch_from_summary(m1, s1, n1, m2, s2, n2):
    re1, re2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = math.sqrt(re1 + re2)
    t = (m1 - m2) / se
    df = (re1 + re2) ** 2 / (re1 ** 2 / (n1 - 1) + re2 ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return t, df, p, se


def main():
    new = {}
    for ds, fname in LOGS.items():
        tests = parse_log(LOG_BASE / fname)
        assert len(tests) == 5, f"{ds}: expected 5 seeds, got {len(tests)}"
        new[ds] = tests

    rows = []
    for ds in PRIMARY:
        metric = PRIMARY[ds]
        vals = np.array([new[ds][s][metric] for s in sorted(new[ds])])
        m_new, s_new = vals.mean(), vals.std(ddof=1)
        om, os_pop = OLD_FLATTEN[ds][metric]
        os_s = os_pop * math.sqrt(5 / 4)  # pop -> sample std
        t, df, p, se = welch_from_summary(m_new, s_new, 5, om, os_s, 5)
        d = (m_new - om) / math.sqrt((s_new ** 2 + os_s ** 2) / 2)
        row = {
            "dataset": ds, "metric": metric,
            "new_mean": m_new, "new_std": s_new,
            "old_flat_mean": om, "old_flat_std_sample": os_s,
            "delta_pp": (m_new - om) * 100, "t": t, "df": df, "p": p,
            "cohens_d": d, "ci95_pp": stats.t.ppf(0.975, df) * se * 100,
            "per_seed": [float(v) for v in vals],
        }
        if ds in OLD_PATCHMEAN and metric in OLD_PATCHMEAN[ds]:
            pm, pm_pop = OLD_PATCHMEAN[ds][metric]
            pm_s = pm_pop * math.sqrt(5 / 4)
            t2, df2, p2, _ = welch_from_summary(m_new, s_new, 5, pm, pm_s, 5)
            row.update({"old_pm_mean": pm, "old_pm_std_sample": pm_s,
                        "delta_pm_pp": (m_new - pm) * 100, "p_pm": p2,
                        "cohens_d_pm": (m_new - pm) / math.sqrt((s_new ** 2 + pm_s ** 2) / 2)})
        rows.append(row)

    p_holm = holm([r["p"] for r in rows])
    for r, ph in zip(rows, p_holm):
        r["p_holm"] = ph
        r["sig"] = sig_stars(ph)

    pm_rows = [r for r in rows if "p_pm" in r]
    pm_holm = holm([r["p_pm"] for r in pm_rows])
    for r, ph in zip(pm_rows, pm_holm):
        r["p_pm_holm"] = ph

    with (OUT / "results.json").open("w") as fh:
        json.dump(rows, fh, indent=2)

    cols = ["dataset", "metric", "new_mean", "new_std", "old_flat_mean", "old_flat_std_sample",
            "delta_pp", "t", "df", "p", "p_holm", "cohens_d", "ci95_pp", "sig",
            "old_pm_mean", "delta_pm_pp", "p_pm", "p_pm_holm"]
    with (OUT / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with (OUT / "results_all_metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "metric", "new_mean", "new_std_sample",
                    "old_flat_mean", "old_flat_popstd", "delta_pp"])
        for ds in PRIMARY:
            for k in ("ba", "pr_auc", "kappa", "roc_auc", "f1"):
                if k not in OLD_FLATTEN[ds]:
                    continue
                vals = np.array([new[ds][s][k] for s in sorted(new[ds])])
                om, op = OLD_FLATTEN[ds][k]
                w.writerow([ds, k, f"{vals.mean():.5f}", f"{vals.std(ddof=1):.5f}",
                            f"{om:.5f}", f"{op:.5f}", f"{(vals.mean() - om) * 100:+.2f}"])

    print(f"{'dataset':16s} {'metric':7s} {'new':>8s} {'old_flat':>9s} {'d_pp':>7s} "
          f"{'p':>8s} {'p_holm':>8s} {'d':>7s} {'ci95':>7s}  sig")
    for r in sorted(rows, key=lambda x: -x["delta_pp"]):
        print(f"{r['dataset']:16s} {r['metric']:7s} {r['new_mean']:.5f} {r['old_flat_mean']:.5f} "
              f"{r['delta_pp']:+7.2f} {r['p']:8.4f} {r['p_holm']:8.4f} {r['cohens_d']:+7.2f} "
              f"{r['ci95_pp']:7.2f}  {r['sig']}")
    print()
    print("patch-mean (exploratory) contrasts:")
    for r in sorted(pm_rows, key=lambda x: -x["delta_pm_pp"]):
        print(f"  {r['dataset']:16s} d_pm={r['delta_pm_pp']:+7.2f}pp  p={r['p_pm']:.4f}  p_holm={r['p_pm_holm']:.4f}")

    # ------------------------------------------------------------- figures ---
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    srt = sorted(rows, key=lambda x: x["delta_pp"])

    # fig-01: delta bar chart (new vs old flatten), sorted, Holm significance
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(srt))
    colors = ["#2b7bba" if r["delta_pp"] >= 0 else "#c0392b" for r in srt]
    ax.barh(y, [r["delta_pp"] for r in srt], color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['dataset']} ({r['metric']})" for r in srt])
    ax.axvline(0, color="k", lw=0.8)
    for i, r in enumerate(srt):
        off = 0.35 if r["delta_pp"] >= 0 else -0.35
        ha = "left" if r["delta_pp"] >= 0 else "right"
        ax.text(r["delta_pp"] + off, i, f"{r['delta_pp']:+.2f}{'' if r['sig']=='ns' else ' '+r['sig']}",
                va="center", ha=ha, fontsize=8)
    ax.set_xlabel("Δ primary metric (new − old flatten), pp")
    ax.set_title("ViT-Small: true cls-token + lr5e-4 + clip1 vs flatten@1e-4\n"
                 "(Holm-corrected Welch test, n=5+5)")
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(FIG / "figure-01-main-comparison.png", dpi=200)
    fig.savefig(FIG / "figure-01-main-comparison.pdf")
    plt.close(fig)

    # fig-02: forest plot with 95% CI
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(srt))
    for i, r in enumerate(srt):
        sig = r["sig"] != "ns"
        ax.errorbar(r["delta_pp"], i, xerr=r["ci95_pp"], fmt="o" if sig else "o",
                    color="#2b7bba" if sig else "#7f8c8d", mfc="white" if not sig else None,
                    ms=5, capsize=3, lw=1.2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['dataset']} ({r['metric']})" for r in srt])
    ax.set_xlabel("Δ primary metric (new − old flatten), pp, 95% CI (Welch)")
    ax.set_title("Effect sizes with 95% confidence intervals\n(filled = Holm-corrected p < .05, open = not significant)")
    fig.tight_layout()
    fig.savefig(FIG / "figure-02-forest-plot.png", dpi=200)
    fig.savefig(FIG / "figure-02-forest-plot.pdf")
    plt.close(fig)

    # fig-03: per-seed grid
    order = sorted(rows, key=lambda x: -x["delta_pp"])
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 9.0))
    rng = np.random.default_rng(42)
    for ax, r in zip(axes.ravel(), order):
        vals = np.array(r["per_seed"])
        x = rng.uniform(-0.10, 0.10, len(vals))
        ax.scatter(x, vals, s=28, color="#2b7bba", zorder=3)
        ax.axhline(r["old_flat_mean"], color="#c0392b", lw=1.4, label="old flatten mean")
        ax.fill_between([-0.25, 0.25],
                        r["old_flat_mean"] - r["old_flat_std_sample"],
                        r["old_flat_mean"] + r["old_flat_std_sample"],
                        color="#c0392b", alpha=0.15, label="old ±1σ (sample)")
        if "old_pm_mean" in r:
            ax.axhline(r["old_pm_mean"], color="#7f8c8d", lw=1.1, ls="--", label="old patch-mean")
        ax.set_title(f"{r['dataset']} — {r['metric']} (Δ {r['delta_pp']:+.2f}pp)", fontsize=9)
        ax.set_xlim(-0.25, 0.25)
        ax.set_xticks([])
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Per-seed test results (new recipe) vs old baselines", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(FIG / "figure-03-per-seed-grid.png", dpi=200)
    fig.savefig(FIG / "figure-03-per-seed-grid.pdf")
    plt.close(fig)

    print("\nfigures written:", sorted(p.name for p in FIG.glob("figure-*")))


if __name__ == "__main__":
    main()
