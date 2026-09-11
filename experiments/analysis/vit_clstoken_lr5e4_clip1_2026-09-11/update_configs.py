#!/usr/bin/env python
"""Update the 10 ViT dataset YAMLs with corrected true cls-token results.

SEED-V and BCIC2020-3 are intentionally left untouched (flatten remains the
recommended head there). Old flatten results and the previously mislabeled
patch-mean ("cls_token") results are preserved under head_comparison.
"""
import re
import sys
from pathlib import Path

import yaml

BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE))
from build_bundle import LOG_BASE, LOGS, OLD_FLATTEN, OLD_PATCHMEAN, parse_log  # noqa: E402

CFG = Path("/data/lz/public/CBraMod/configs/backbones/vit_small_dinov3")

RUNS = {
    "CHB-MIT": "vit_small_dinov3_chb_mit_p4_clstoken_lr5e4_clip1_5seed_v1",
    "TUAB": "vit_small_dinov3_tuab_p4_clstoken_lr5e4_aug_5seed_v1",
    "TUEV": "vit_small_dinov3_tuev_p4_clstoken_lr5e4_clip1_5seed_v1",
    "ISRUC": "vit_small_dinov3_isruc_p12_clstoken_lr5e4_clip1_5seed_v1",
    "FACED": "vit_small_dinov3_faced_p2_clstoken_lr5e4_clip1_5seed_v1",
    "PhysioNet-MI": "vit_small_dinov3_physionet_mi_p1_clstoken_lr5e4_clip1_5seed_v1",
    "SHU-MI": "vit_small_dinov3_shu_mi_p2_clstoken_lr5e4_clip1_5seed_v1",
    "Mumtaz2016": "vit_small_dinov3_mumtaz_p4_clstoken_lr5e4_clip1_5seed_v1",
    "MentalArithmetic": "vit_small_dinov3_mentalarithmetic_p4_clstoken_lr5e4_clip1_5seed_v1",
    "HMC": "vit_small_dinov3_hmc_p16_clstoken_lr5e4_clip1_5seed_v1",
}

EXTRA_RESULTS = {
    "TUAB": ["  augmentation: [mirror, time_roll, amplitude_scale]"],
}

PM_DATASETS = {"CHB-MIT", "TUAB", "ISRUC", "FACED", "SHU-MI", "PhysioNet-MI", "MentalArithmetic"}

FLATTEN_NOTES = {
    "TUAB": [
        "flatten head (51072-dim FC) overfits; val peaks at epoch 1 then declines.",
        "Superseded by the compact heads (patch-mean then true CLS token).",
    ],
}

PER_SEED_OLD = {
    "Mumtaz2016": [
        "    per_seed:",
        "      42: {epoch: 3, ba: 0.92693, pr_auc: 0.98055, roc_auc: 0.97803}",
        "      43: {epoch: 8, ba: 0.91315, pr_auc: 0.98150, roc_auc: 0.97885}",
        "      44: {epoch: 2, ba: 0.88050, pr_auc: 0.97761, roc_auc: 0.97536}",
        "      45: {epoch: 8, ba: 0.91064, pr_auc: 0.98163, roc_auc: 0.97997}",
        "      46: {epoch: 5, ba: 0.90485, pr_auc: 0.98205, roc_auc: 0.98177}",
    ],
    "HMC": [
        "    per_seed:",
        "      seed42: {ba: 0.74411, kappa: 0.69731, f1: 0.76439}",
        "      seed43: {ba: 0.74664, kappa: 0.69740, f1: 0.76357}",
        "      seed44: {ba: 0.74453, kappa: 0.69586, f1: 0.76363}",
        "      seed45: {ba: 0.75279, kappa: 0.70012, f1: 0.76501}",
        "      seed46: {ba: 0.74913, kappa: 0.69957, f1: 0.76480}",
    ],
}

NOTES = {
    "CHB-MIT": [
        "True CLS-token head at lr 5e-4 with clip 1, replacing the flatten head",
        "(+1.93pp pr_auc; BA -0.74pp). Balanced sampling and P=4 retained. Five seeds,",
        "validation-selected checkpoint, one final test per seed.",
    ],
    "TUAB": [
        "True CLS-token head at lr 5e-4 with clip 1; mirror/time-roll/amplitude-scale",
        "augmentation retained. At parity with the mislabeled patch-mean 'gap' recipe",
        "and above the earlier flatten head (+2.48pp pr_auc). Five seeds, validation-selected",
        "checkpoint, one final test per seed.",
    ],
    "TUEV": [
        "True CLS-token head at lr 5e-4 with clip 1, replacing the flatten head",
        "(+1.65pp kappa, raw p=.057). The previously pending patch-mean comparison is",
        "superseded by the corrected CLS-token semantics. Five seeds, validation-selected",
        "checkpoint, one final test per seed.",
    ],
    "ISRUC": [
        "True CLS-token head at lr 5e-4 with clip 1, replacing the flatten head",
        "(+0.90pp kappa). The [B,20,C,T] loader contract and P=12 are retained. Five seeds,",
        "validation-selected checkpoint, one final test per seed.",
    ],
    "FACED": [
        "True CLS-token head at lr 5e-4 with clip 1, replacing the flatten head",
        "(+6.47pp kappa) and reversing the earlier mislabeled patch-mean comparison.",
        "Five seeds, validation-selected checkpoint, one final test per seed.",
    ],
    "PhysioNet-MI": [
        "True CLS-token head at lr 5e-4 with clip 1, replacing the flatten head",
        "(+8.06pp kappa; all five seeds above the old mean). Native P=1 geometry retained.",
        "Five seeds, validation-selected checkpoint, one final test per seed.",
    ],
    "SHU-MI": [
        "True CLS-token head at lr 5e-4 with clip 1, replacing the flatten head",
        "(+2.72pp pr_auc; one outlier seed lowers the aggregate). clip=1024 and scale=32",
        "are retained. Five seeds, validation-selected checkpoint, one final test per seed.",
    ],
    "Mumtaz2016": [
        "True CLS-token head at lr 5e-4 with clip 1; parity with the flatten head",
        "(-0.09pp pr_auc). Amplitude-scale augmentation retained. Five seeds,",
        "validation-selected checkpoint, one final test per seed.",
    ],
    "MentalArithmetic": [
        "True CLS-token head at lr 5e-4 with clip 1, replacing the flatten head",
        "(+15.10pp pr_auc; high seed variance). Five seeds, validation-selected checkpoint,",
        "one final test per seed.",
    ],
    "HMC": [
        "True CLS-token head at lr 5e-4 with clip 1; parity with the flatten head",
        "(+0.06pp kappa). P=16 retained. Five seeds, validation-selected checkpoint,",
        "one final test per seed.",
    ],
}


def fmt_pairs(pairs):
    return "{" + ", ".join(f"{k}: {v:.5f}" for k, v in pairs) + "}"


def compute(text):
    tests = parse_log(LOG_BASE / LOGS[text])
    assert len(tests) == 5, text
    metrics = list(OLD_FLATTEN[text].keys())
    mean = [(k, sum(tests[s][k] for s in tests) / len(tests)) for k in metrics]
    std = [
        (k, (sum((tests[s][k] - dict(mean)[k]) ** 2 for s in tests) / len(tests)) ** 0.5)
        for k in metrics
    ]
    return mean, std


def build_tail(ds):
    new_mean, new_std = compute(ds)
    lines = [
        "results:",
        "  protocol: validation_selected_checkpoint_once_on_test",
        "  feature_aggregation: cls_token",
        "  lr: 0.0005",
        "  clip_value: 1.0",
    ]
    lines += EXTRA_RESULTS.get(ds, [])
    lines += [
        "  seeds: [42, 43, 44, 45, 46]",
        f"  mean: {fmt_pairs(new_mean)}",
        f"  std_population: {fmt_pairs(new_std)}",
        "head_comparison:",
        "  flatten:",
        "    status: complete",
        "    feature_aggregation: flatten",
        "    lr: 0.0001",
        "    seeds: [42, 43, 44, 45, 46]",
    ]
    lines += PER_SEED_OLD.get(ds, [])
    flat_mean = [(k, v[0]) for k, v in OLD_FLATTEN[ds].items()]
    flat_std = [(k, v[1]) for k, v in OLD_FLATTEN[ds].items()]
    lines += [
        f"    mean: {fmt_pairs(flat_mean)}",
        f"    std_population: {fmt_pairs(flat_std)}",
    ]
    if ds in FLATTEN_NOTES:
        lines.append("    note: >-")
        lines += ["      " + s for s in FLATTEN_NOTES[ds]]
    if ds in PM_DATASETS:
        lines += [
            "  patch_mean:",
            "    status: complete",
            "    feature_aggregation: gap",
            "    note: >-",
            "      Previously recorded as \"cls_token\"; verified to be patch-token mean",
            "      pooling (timm Eva, global_pool='avg'), not the CLS token.",
            "    seeds: [42, 43, 44, 45, 46]",
        ]
        pm_mean = [(k, v[0]) for k, v in OLD_PATCHMEAN[ds].items()]
        pm_std = [(k, v[1]) for k, v in OLD_PATCHMEAN[ds].items()]
        lines += [
            f"    mean: {fmt_pairs(pm_mean)}",
            f"    std_population: {fmt_pairs(pm_std)}",
        ]
    lines += ["  recommendation: cls_token", "notes: >-"]
    lines += ["  " + s for s in NOTES[ds]]
    lines += [
        "output:",
        f"  model_root: experiments/checkpoints/{RUNS[ds]}/seed{{seed}}",
        f"  log_root: experiments/logs/{RUNS[ds]}/seed{{seed}}",
    ]
    return "\n".join(lines) + "\n"


def update_one(ds):
    path = CFG / f"{ds}.yaml"
    text = path.read_text()
    head, sep, _ = text.partition("\nresults:")
    assert sep, ds
    head = re.sub(r"^  feature_aggregation: (flatten|gap)$", "  feature_aggregation: cls_token",
                  head, count=1, flags=re.M)
    head = re.sub(r"^  lr: 0\.0001$", "  lr: 0.0005", head, count=1, flags=re.M)
    head = re.sub(r"^  clip_value: -1\.0$", "  clip_value: 1.0", head, count=1, flags=re.M)
    new_text = head + "\n" + build_tail(ds)
    parsed = yaml.safe_load(new_text)
    assert parsed["vision"]["feature_aggregation"] == "cls_token"
    assert parsed["training"]["lr"] == 0.0005
    assert parsed["training"]["clip_value"] == 1.0
    assert parsed["results"]["feature_aggregation"] == "cls_token"
    assert parsed["head_comparison"]["recommendation"] == "cls_token"
    path.write_text(new_text)
    r = parsed["results"]
    print(f"{ds:16s} results={r['mean']} std={r['std_population']}")


def main():
    for ds in RUNS:
        update_one(ds)
    print(f"\nupdated {len(RUNS)} configs; SEED-V and BCIC2020-3 untouched")


if __name__ == "__main__":
    main()
