#!/usr/bin/env python
"""Backfill the foundation rerun v2 results into the per-dataset configs.

Reads the v2 run logs + checkpoint tree (the layout the v2 queue writes),
aggregates completed seeds per (model, dataset) exactly like
collect_foundation_results.py (validation-selected checkpoint, one final test
per seed), and rewrites `configs/foundation_models/<model>/<DATASET>.yaml` in
place as the v2 record: the resolved training hyperparameters (parsed from the
run command line), metric mean/std, per-seed epoch + metrics, notes, and the
v2 output roots.  The previous (v1) primary mean/std is cited in the new notes
(read from the pre-backfill snapshot when present).

The four multi-dataset `v2_*.yaml` files stay pure runnable configs and are
intentionally NOT written here; records live only in the per-dataset files.

Dry-run by default: prints a per-dataset summary.  Pass --write_datasets to
apply.
"""
import argparse
import shlex
import statistics
import sys
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))

from collect_foundation_results import PUBLISHED, REVE_FALLBACK_DATASETS, SAFE_TO_DATASET, parse_run, read_text  # noqa: E402
from configs.foundation import FOUNDATION_SPECS  # noqa: E402

DEFAULT_LOG_ROOT = 'experiments/logs/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1'
DEFAULT_CHECKPOINT_ROOT = 'experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1'
PREVIOUS_SNAPSHOT_ROOT = DEFAULT_LOG_ROOT + '/pre_backfill_yaml/datasets_v1'
DEFAULT_YAML_ROOT = 'configs/foundation_models'
V2_RUN_ROOT = 'foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1'
MODEL_LABELS = {'cbramod': 'CBraMod', 'reve': 'REVE-Base'}
TASK_DESC = {
    'FACED': '9-class emotion recognition',
    'TUAB': 'binary abnormal EEG detection',
    'HMC': '5-class sleep staging',
    'Mumtaz2016': 'binary MDD detection',
    'PhysioNet-MI': '4-class motor imagery',
    'CHB-MIT': 'binary seizure detection',
    'TUEV': '6-class event type classification',
    'ISRUC': '5-class sleep staging',
    'SEED-V': '5-class emotion recognition',
    'SHU-MI': 'binary motor imagery',
    'BCIC2020-3': '5-class imagined speech',
    'MentalArithmetic': 'binary mental workload',
}
DEVIATIONS = {
    'cbramod': 'The released multi-LR schedule is not used; the whole network '
               'trains with a single 1e-4 learning rate (documented deviation).',
    'reve': 'The released lp -> ft warm start, mixup, StableAdamW, LoRA and '
            'model souping are not applied (documented deviation).',
}
RECORD_NOTES = {
    ('reve', 'CHB-MIT'): 'CHB-MIT is not part of the released REVE benchmark; the input scale '
                         '(microvolt / 100) and the pooled-token readout are fallback settings and this '
                         'record is reference-only.',
    ('reve', 'SEED-V'): 'SEED-V is not part of the released REVE benchmark; the input scale '
                        '(microvolt / 100) is a fallback setting and this record is reference-only. '
                        'CB1/CB2 map to the inferior occipital OI1h/OI2h positions because the released '
                        'position bank has no cerebellar entries.',
    ('reve', 'SHU-MI'): 'SHU-MI is not part of the released REVE benchmark; the input scale '
                        '(microvolt / 100) and the pooled-token readout are fallback settings and this '
                        'record is reference-only.',
    ('reve', 'TUEV'): 'The TUEV loader keeps microvolt / 100, the stored-array equivalent of the released '
                      'volt-scale memmap factor x1e4 already documented for TUAB.',
    ('reve', 'ISRUC'): 'REVE ISRUC runs at batch 8 instead of the configured batch size 16 because the '
                       '22-layer encoder exceeds the GPU memory at batch 16 (CUDA OOM); this is the only '
                       'batch-size deviation in the sweep.',
    ('reve', 'BCIC2020-3'): 'The released Speech config selects the pooled-token (last) readout, but on the '
                            'stored arrays that readout stays at chance under the campaign recipe (five-seed '
                            'kappa 0.107); this record uses the non-pooling readout instead (flatten all patch '
                            'tokens together with the context token), which the released REVE paper also uses for '
                            'its probe columns.',
    ('cbramod', 'TUEV'): 'TUEV validation kappa peaks at the first epoch for every seed (the validation '
                         'split is about 89% majority class), so all selected checkpoints are epoch 1.',
}
REMAINING_DATASETS = {'CHB-MIT', 'TUEV', 'ISRUC', 'SEED-V', 'SHU-MI', 'BCIC2020-3', 'MentalArithmetic'}
APPENDIX_POINTERS = {
    'remaining': 'experiments/reports/foundation_rerun_v2_remaining_appendix.md',
    'main': 'experiments/reports/foundation_rerun_v2_appendix.md',
}

TRAINING_KEYS = (
    'lr', 'batch_size', 'epochs', 'weight_decay', 'min_lr', 'warmup_epochs',
    'warmup_start_factor', 'clip_value', 'ema_decay', 'num_workers', 'optimizer',
    'label_smoothing', 'binary_pos_weight', 'dropout', 'drop_path_rate', 'early_stop',
    'frozen', 'multi_lr', 'use_pretrained_weights', 'balanced_sampling',
    'balanced_sampling_power', 'balanced_sampling_min_share',
    'balanced_sampling_negative_ratio', 'mirror_augmentation', 'mirror_prob',
    'time_roll_augmentation', 'time_roll_prob', 'time_roll_max_fraction',
    'amplitude_scale_augmentation', 'mixup_augmentation', 'amp', 'amp_dtype',
    'test_each_epoch', 'run_final_test', 'selection_metric',
)


def coerce_value(text):
    if text == 'True':
        return True
    if text == 'False':
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_training(text):
    first_line = text.splitlines()[0] if text.splitlines() else ''
    tokens = shlex.split(first_line)
    raw = {}
    for index, token in enumerate(tokens):
        if token.startswith('--') and index + 1 < len(tokens) and not tokens[index + 1].startswith('--'):
            raw[token[2:]] = tokens[index + 1]
    return {key: coerce_value(raw[key]) for key in TRAINING_KEYS if key in raw}


def format_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        text = repr(value)
        mantissa, _, exponent = text.partition('e')
        if exponent and '.' not in mantissa:
            text = mantissa + '.0e' + exponent  # YAML resolvers need a dot for e-notation floats
        return text
    return str(value)


def loader_description(model, dataset):
    if model == 'cbramod':
        return 'the loader applies the released CBraMod normalization (microvolt / 100, no clipping)'
    spec = FOUNDATION_SPECS[dataset]['reve']
    if dataset in REVE_FALLBACK_DATASETS:
        return ('the loader applies a fallback microvolt / {:g} without clipping (this dataset is not part '
                'of the released REVE benchmark) and supplies position-bank coordinates for {} channels').format(
                    spec['input_scale'], len(spec['electrodes']))
    return ('the loader applies the released REVE scale_factor (microvolt / {:g}, no clipping) '
            'and supplies released electrode positions ({} channels)').format(
                spec['input_scale'], len(spec['electrodes']))


def read_previous_record(snapshot_path, target_path, primary):
    for path in (snapshot_path, target_path):
        if not path.is_file():
            continue
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        if 'Foundation rerun v2 record' in str(cfg.get('notes', '')):
            continue
        entry = (cfg.get('results') or {}).get(primary)
        if isinstance(entry, dict) and 'mean' in entry and 'std' in entry:
            return float(entry['mean']), float(entry['std'])
    return None


def render_dataset_record(model, dataset, entries, training, previous, run_root):
    seeds = sorted(entries)
    metric_names = [name for name in entries[seeds[0]] if name != 'epoch']
    primary = metric_names[1]
    primary_values = [entries[seed][primary] for seed in seeds]
    primary_mean = statistics.mean(primary_values)

    lines = ['dataset: {}'.format(dataset), 'model_arch: {}'.format(model), 'training:']
    for key in TRAINING_KEYS:
        if key in training:
            lines.append('  {}: {}'.format(key, format_value(training[key])))
    lines.append('protocol:')
    lines.append('  seeds:')
    for seed in seeds:
        lines.append('  - {}'.format(seed))
    lines.append('  selection: validation_selected_checkpoint_once_on_test')
    lines.append('results:')
    for name in metric_names:
        values = [entries[seed][name] for seed in seeds]
        lines.append('  {}:'.format(name))
        lines.append('    mean: {}'.format(round(statistics.mean(values), 5)))
        lines.append('    std: {}'.format(round(statistics.pstdev(values), 5)))
    lines.append('  per_seed:')
    for seed in seeds:
        lines.append('    {}:'.format(seed))
        for name in ('epoch',) + tuple(metric_names):
            lines.append('      {}: {}'.format(name, round(entries[seed][name], 5)))

    notes = [
        'Foundation rerun v2 record for {} on {} ({}).'.format(MODEL_LABELS[model], dataset, TASK_DESC[dataset]),
        'Recipe: the v2 unified recipe (lr 1e-4, warmup 3 epochs at factor 0.1, EMA 0.995, '
        'weight decay 5e-4, gradient clipping 1.0, early stop 10) with {} epochs and {} checkpoint '
        'selection from configs/downstream.py; {}.'.format(
            training.get('epochs'), training.get('selection_metric'), loader_description(model, dataset)),
        DEVIATIONS[model],
        'Validation-selected checkpoint, one final test per seed, seeds 42-46, population std.',
    ]
    record_note = RECORD_NOTES.get((model, dataset))
    if record_note:
        notes.append(record_note)
    published = PUBLISHED.get((dataset, model), {}).get('PR_AUC' if primary == 'pr_auc' else primary)
    if published:
        published_text = '{:.4f}'.format(published[0])
        if published[1] is not None:
            published_text += ' +/- {:.4f}'.format(published[1])
        notes.append('Published reference {} {} (delta {:+.4f}).'.format(
            primary, published_text, primary_mean - published[0]))
    if previous:
        notes.append('Replaces the v1 matched-pipeline record ({} {:.4f} +/- {:.4f}).'.format(
            primary, previous[0], previous[1]))
    notes.append('See {}.'.format(
        APPENDIX_POINTERS['remaining'] if dataset in REMAINING_DATASETS else APPENDIX_POINTERS['main']))
    lines.append('notes: >-')
    lines.extend(textwrap.fill(' '.join(notes), width=78, initial_indent='  ', subsequent_indent='  ',
                               break_on_hyphens=False).split('\n'))
    lines.append('output:')
    lines.append('  model_root: experiments/checkpoints/' + run_root + '/' + model + '/seed{seed}')
    lines.append('  log_root: experiments/logs/' + run_root + '/' + model + '/seed{seed}')
    return '\n'.join(lines) + '\n'


def collect_groups(log_root, checkpoint_root):
    latest = {}
    for log_path in log_root.glob('*/seed*/vision/*/*.log'):
        parts = log_path.parts
        model, seed = parts[-5], parts[-4]
        dataset = SAFE_TO_DATASET.get(parts[-2], parts[-2])
        key = (model, dataset, seed)
        if key not in latest or log_path.name > latest[key].name:
            latest[key] = log_path

    groups = {}
    trainings = {}
    for (model, dataset, seed), log_path in sorted(latest.items()):
        parsed = parse_run(log_path)
        if 'test_primary' not in parsed:
            continue
        checkpoint = checkpoint_root / model / seed / log_path.parts[-2] / 'best.pth'
        if not checkpoint.is_file():
            continue
        entry = {'epoch': parsed['best_epoch'], 'ba': float(parsed['test_ba'])}
        entry[parsed['test_primary_name']] = float(parsed['test_primary'])
        entry[parsed['test_secondary_name']] = float(parsed['test_secondary'])
        groups.setdefault((model, dataset), {})[int(seed.replace('seed', ''))] = entry
        trainings.setdefault((model, dataset), parse_training(read_text(log_path)))
    return groups, trainings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root', default=DEFAULT_LOG_ROOT)
    parser.add_argument('--checkpoint_root', default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument('--yaml_root', default=DEFAULT_YAML_ROOT)
    parser.add_argument('--run_root', default=V2_RUN_ROOT,
                        help='campaign run root recorded in the output block')
    parser.add_argument('--write_datasets', action='store_true',
                        help='rewrite the per-dataset record files under <yaml_root>/<model>/<DATASET>.yaml')
    args = parser.parse_args()

    log_root = ROOT / args.log_root
    checkpoint_root = ROOT / args.checkpoint_root
    yaml_root = ROOT / args.yaml_root

    groups, trainings = collect_groups(log_root, checkpoint_root)
    if not groups:
        print('no completed runs found under {}'.format(log_root))
        return

    print('=== per-dataset records ===')
    for (model, dataset), entries in sorted(groups.items()):
        path = yaml_root / model / '{}.yaml'.format(dataset)
        seeds = sorted(entries)
        metric_names = [name for name in entries[seeds[0]] if name != 'epoch']
        primary = metric_names[1]
        values = [entries[seed][primary] for seed in seeds]
        snapshot_path = ROOT / PREVIOUS_SNAPSHOT_ROOT / model / '{}.yaml'.format(dataset)
        previous = read_previous_record(snapshot_path, path, primary)
        previous_text = 'no previous record' if previous is None else 'prev v1 {:.4f} +/- {:.4f}'.format(
            previous[0], previous[1])
        print('  {model}-{dataset}: {n} seeds, {primary} {mean:.5f} +/- {std:.5f} ({previous})'.format(
            model=model, dataset=dataset, n=len(seeds), primary=primary,
            mean=statistics.mean(values), std=statistics.pstdev(values), previous=previous_text))
        if args.write_datasets:
            record = render_dataset_record(model, dataset, entries, trainings.get((model, dataset)), previous, args.run_root)
            path.write_text(record, encoding='utf-8')
            print('    -> dataset record written')


if __name__ == '__main__':
    main()
