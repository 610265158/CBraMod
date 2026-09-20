#!/usr/bin/env python
"""End-to-end verification of the foundation rerun v2 (clip 1.0) campaign.

Part A - data pipeline: split counts, sample shapes/labels, and the numerical
equivalence loader(view) == raw / released_scale for the five datasets
(FACED, TUAB, HMC, Mumtaz2016, PhysioNet-MI) at the scales the v2 configs use.

Part B - run records: for each completed run, cross-check the log command line
(model arch, dataset dir, lr/epochs/selection, warmup/EMA/weight decay/clip,
single final test), the printed normalization scale, the test metrics against
the results CSV, and the selected-epoch checkpoint.

Part C - config records: the per-dataset record files
(configs/foundation_models/<model>/<DATASET>.yaml) carry the v2 training block,
results and output roots and match the CSV. Each record is also a complete
runnable config, so the multi-dataset launcher YAMLs were removed.

Part D - queue integrity: chain log present, no FAILED runs, finished marker,
and 50 .done sentinels.

Run after the v2 queue finishes and after collect_foundation_results.py and
write_foundation_rerun_v2_results.py --write_datasets have produced the records.
Exits non-zero when any check fails; prints only failures plus a summary.
"""
import argparse
import csv
import os
import pickle
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

V2_LOGS = 'experiments/logs/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1'
V2_CKPTS = 'experiments/checkpoints/foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1'
DEFAULT_CSV = 'experiments/reports/foundation_rerun_v2_results.csv'
QUEUE_LOG = V2_LOGS + '/chain_v2_main.log'
EXPECTED_RUNS = 50
EXPECTED_SEEDS_PER_GROUP = 5
TUAB_ONLY = {'TUAB'}

CHECKS = []
FAILURES = []


def check(name, ok, detail=''):
    CHECKS.append(name)
    if not ok:
        FAILURES.append('{} {}'.format(name, '({})'.format(detail) if detail else ''))


EXPECTED_SCALE = {
    ('cbramod', 'FACED'): '100.0',
    ('cbramod', 'TUAB'): '100.0',
    ('cbramod', 'HMC'): '100.0',
    ('cbramod', 'Mumtaz2016'): '100.0',
    ('cbramod', 'PhysioNet-MI'): '100.0',
    ('reve', 'FACED'): '1000.0',
    ('reve', 'TUAB'): '100.0',
    ('reve', 'HMC'): '10.0',
    ('reve', 'Mumtaz2016'): '100.0',
    ('reve', 'PhysioNet-MI'): '100.0',
}
EXPECTED_EPOCHS = {'FACED': '50', 'TUAB': '10', 'HMC': '30', 'Mumtaz2016': '30',
                   'PhysioNet-MI': '30'}
EXPECTED_SELECTION = {'FACED': 'kappa', 'TUAB': 'pr_auc', 'HMC': 'kappa',
                      'Mumtaz2016': 'pr_auc', 'PhysioNet-MI': 'kappa'}
EXPECTED_DIR = {
    'FACED': 'faced/processed',
    'TUAB': 'TUAB',
    'HMC': 'haaglanden',
    'Mumtaz2016': 'MDDPHCED',
    'PhysioNet-MI': 'eeg-motor',
}
EXPECTED_COUNTS = {
    'FACED': {'train': 6720, 'val': 1680, 'test': 1932},
    'TUAB': {'train': 297103, 'val': 75407, 'test': 36945},
    'HMC': {'train': 91248, 'val': 22124, 'test': 23871},
    'Mumtaz2016': {'train': 4891, 'val': 1041, 'test': 1211},
    'PhysioNet-MI': {'train': 6300, 'val': 1734, 'test': 1803},
}
BIG = ROOT.parent / 'BigDownstream'
HMC_DIR = BIG / 'haaglanden-medisch-centrum-sleep-staging-database-1.1' / 'processed'
MUMTAZ_DIR = BIG / 'MDDPHCED' / 'processed_lmdb_75hz'
PHYSIO_DIR = BIG / 'eeg-motor-movementimagery-dataset-1.0.0'


def lmdb_keys(path):
    import lmdb
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    with env.begin(write=False) as txn:
        keys = pickle.loads(txn.get(b'__keys__'))
    env.close()
    return keys


def lmdb_raw(path, key):
    import lmdb
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    with env.begin(write=False) as txn:
        pair = pickle.loads(txn.get(key.encode()))
    env.close()
    return pair['sample']


def check_collate(name, dataset, expected_shape):
    x, _ = dataset.collate([dataset[0], dataset[1]])
    shape = tuple(x.shape)
    check('{} collated shape {}'.format(name, expected_shape), shape == (2,) + expected_shape, str(shape))


def check_lmdb_counts(name, path):
    counts = {key: len(value) for key, value in lmdb_keys(path).items()}
    check('{} split counts'.format(name), counts == EXPECTED_COUNTS[name], str(counts))
    return counts


def data_checks():
    from datasets import shape_utils as su

    # FACED (LMDB, 32 channels, /100 or /1000)
    faced_dir = BIG / 'faced' / 'processed'
    faced_keys = lmdb_keys(faced_dir)
    counts = {key: len(value) for key, value in faced_keys.items()}
    check('FACED split counts', counts == EXPECTED_COUNTS['FACED'], str(counts))
    from datasets.faced_dataset import CustomDataset as FacedDS
    for scale in (100.0, 1000.0):
        su.configure_eeg_normalization(limit=float('inf'), scale=scale)
        x, y = FacedDS(str(faced_dir), mode='train')[0]
        raw = lmdb_raw(faced_dir, faced_keys['train'][0])
        expected = np.asarray(raw, dtype=np.float32) / np.float32(scale)
        check('FACED numeric raw/{}'.format(scale), np.allclose(x, expected))
        check('FACED finite'.format(), bool(np.isfinite(x).all()))
        check('FACED label 0-8', 0 <= int(y) <= 8, str(y))
    check_collate('FACED', FacedDS(str(faced_dir), mode='train'), (32, 2000))

    # TUAB (pkl split, 16 bipolar channels, /100)
    tuab_dir = BIG / 'TUAB'
    for split, expected_count in EXPECTED_COUNTS['TUAB'].items():
        n = len(os.listdir(tuab_dir / split))
        check('TUAB {} count'.format(split), n == expected_count, str(n))
    from datasets.tuab_dataset import CustomDataset as TuabDS
    su.configure_eeg_normalization(limit=float('inf'), scale=100.0)
    x, y = TuabDS(str(tuab_dir), mode='train')[0]
    train_files = [str(tuab_dir / 'train' / name) for name in os.listdir(tuab_dir / 'train')]
    raw = pickle.load(open(train_files[0], 'rb'))['X']
    check('TUAB numeric raw/100', np.allclose(x, np.asarray(raw, dtype=np.float32) / np.float32(100.0)))
    check('TUAB finite', bool(np.isfinite(x).all()))
    check('TUAB label in {0,1}', int(y) in (0, 1), str(y))
    check_collate('TUAB', TuabDS(str(tuab_dir), mode='train'), (16, 2000))

    # HMC (pkl split, 4 channels, /10 or /100)
    for split, expected_count in EXPECTED_COUNTS['HMC'].items():
        n = len([f for f in os.listdir(HMC_DIR / split) if f.endswith('.pkl')])
        check('HMC {} count'.format(split), n == expected_count, str(n))
    from datasets.hmc_dataset import CustomDataset as HmcDS
    import glob
    for scale in (10.0, 100.0):
        su.configure_eeg_normalization(limit=float('inf'), scale=scale)
        x, y = HmcDS(str(HMC_DIR), mode='train')[0]
        first = sorted(glob.glob(str(HMC_DIR / 'train' / '*.pkl')))[0]
        raw = pickle.load(open(first, 'rb'))['X']
        check('HMC numeric raw/{}'.format(scale), np.allclose(x, np.asarray(raw, dtype=np.float32) / np.float32(scale)))
        check('HMC finite', bool(np.isfinite(x).all()))
        check('HMC label 0-4', 0 <= int(y) <= 4, str(y))
    check_collate('HMC', HmcDS(str(HMC_DIR), mode='train'), (4, 6000))

    # Mumtaz2016 (LMDB, [19,5,200] stored -> collated [B,19,1000], /100)
    mumtaz_keys = lmdb_keys(MUMTAZ_DIR)
    counts = {key: len(value) for key, value in mumtaz_keys.items()}
    check('Mumtaz2016 split counts', counts == EXPECTED_COUNTS['Mumtaz2016'], str(counts))
    from datasets.mumtaz_dataset import CustomDataset as MumtazDS
    su.configure_eeg_normalization(limit=float('inf'), scale=100.0)
    x, y = MumtazDS(str(MUMTAZ_DIR), mode='train')[0]
    raw = lmdb_raw(MUMTAZ_DIR, mumtaz_keys['train'][0])
    check('Mumtaz2016 numeric raw/100', np.allclose(x, np.asarray(raw, dtype=np.float32) / np.float32(100.0)))
    check('Mumtaz2016 finite', bool(np.isfinite(x).all()))
    check('Mumtaz2016 label in {0,1}', int(y) in (0, 1), str(y))
    check_collate('Mumtaz2016', MumtazDS(str(MUMTAZ_DIR), mode='train'), (19, 1000))

    # PhysioNet-MI (LMDB, [64,4,200] stored -> collated [B,64,800], /100)
    physio_keys = lmdb_keys(PHYSIO_DIR)
    counts = {key: len(value) for key, value in physio_keys.items()}
    check('PhysioNet-MI split counts', counts == EXPECTED_COUNTS['PhysioNet-MI'], str(counts))
    from datasets.physio_dataset import CustomDataset as PhysioDS
    su.configure_eeg_normalization(limit=float('inf'), scale=100.0)
    x, y = PhysioDS(str(PHYSIO_DIR), mode='train')[0]
    raw = lmdb_raw(PHYSIO_DIR, physio_keys['train'][0])
    check('PhysioNet-MI numeric raw/100', np.allclose(x, np.asarray(raw, dtype=np.float32) / np.float32(100.0)))
    check('PhysioNet-MI finite', bool(np.isfinite(x).all()))
    check('PhysioNet-MI label 0-3', 0 <= int(y) <= 3, str(y))
    check_collate('PhysioNet-MI', PhysioDS(str(PHYSIO_DIR), mode='train'), (64, 800))

    su.reset_eeg_normalization()


def run_checks(csv_path):
    csv_file = Path(csv_path)
    if not csv_file.is_absolute():
        csv_file = ROOT / csv_file
    check('results CSV exists', csv_file.is_file(), str(csv_file))
    rows = []
    if csv_file.is_file():
        rows = [r for r in csv.DictReader(csv_file.open(encoding='utf-8')) if r['status'] == 'done']
    check('completed runs == {}'.format(EXPECTED_RUNS), len(rows) == EXPECTED_RUNS, str(len(rows)))

    queue_path = ROOT / QUEUE_LOG
    queue_text = queue_path.read_text(errors='replace') if queue_path.is_file() else ''
    check('chain log exists', queue_path.is_file())
    check('chain log has no FAILED', 'FAILED' not in queue_text)
    check('queue finished marker', 'foundation v2 queue finished' in queue_text)
    sentinels = list((ROOT / V2_CKPTS).rglob('.done'))
    check('{} .done sentinels'.format(EXPECTED_RUNS), len(sentinels) == EXPECTED_RUNS, str(len(sentinels)))

    for r in rows:
        model, dataset, seed = r['model'], r['dataset'], r['seed']
        tag = '{} {} {}'.format(model, dataset, seed)
        log_path = ROOT / r['log']
        if not log_path.is_file():
            check(tag + ' log exists', False, str(log_path))
            continue
        text = log_path.read_text(errors='replace').replace('\r', '')

        check(tag + ' cmd model_arch', '--model_arch {}'.format(model) in text)
        check(tag + ' cmd datasets_dir', EXPECTED_DIR[dataset] in text)
        check(tag + ' cmd lr', '--lr 0.0001' in text)
        check(tag + ' cmd epochs', '--epochs {}'.format(EXPECTED_EPOCHS[dataset]) in text)
        check(tag + ' cmd selection', '--selection_metric {}'.format(EXPECTED_SELECTION[dataset]) in text)
        check(tag + ' cmd warmup 3', '--warmup_epochs 3' in text)
        check(tag + ' cmd ema 0.995', '--ema_decay 0.995' in text)
        check(tag + ' cmd wd 5e-4', '--weight_decay 0.0005' in text)
        check(tag + ' cmd clip 1.0', '--clip_value 1.0' in text)
        check(tag + ' cmd test_each_epoch False', '--test_each_epoch False' in text)
        check(tag + ' loader scale', 'EEG loader normalization: clip disabled, scale {}'.format(
            EXPECTED_SCALE[(model, dataset)]) in text)
        n_test = text.count('Test Evaluation:')
        check(tag + ' exactly one final test', n_test == 1, str(n_test))
        check(tag + ' no traceback', 'Traceback' not in text)

        match = re.search(r'Test Evaluation: ba: ([\d.]+), (?:kappa|pr_auc): ([\d.]+)', text)
        check(tag + ' test line parsed', match is not None)
        if match:
            check(tag + ' log==csv ba', abs(float(match.group(1)) - float(r['test_ba'])) < 1e-6)
            check(tag + ' log==csv primary', abs(float(match.group(2)) - float(r['test_primary'])) < 1e-6)

        checkpoint_dir = (ROOT / r['checkpoint']).parent
        check(tag + ' best.pth exists', (checkpoint_dir / 'best.pth').is_file())
        check(tag + ' selected-epoch checkpoint',
              len(list(checkpoint_dir.glob('epoch{}_*.pth'.format(r['best_epoch'])))) >= 1)
    return rows


def dataset_file_checks(rows, yaml_root):
    for model in ('cbramod', 'reve'):
        for dataset in ('FACED', 'HMC', 'Mumtaz2016', 'PhysioNet-MI', 'TUAB'):
            path = yaml_root / model / '{}.yaml'.format(dataset)
            tag = 'dataset-file {}-{}'.format(model, dataset)
            check(tag + ' exists', path.is_file(), str(path))
            if not path.is_file():
                continue
            cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
            training = cfg.get('training')
            check(tag + ' has training block', isinstance(training, dict) and len(training) >= 30)
            if isinstance(training, dict):
                check(tag + ' training lr', float(training.get('lr', -1.0)) == 0.0001)
                check(tag + ' training epochs', str(training.get('epochs')) == EXPECTED_EPOCHS[dataset])
                check(tag + ' training selection',
                      training.get('selection_metric') == EXPECTED_SELECTION[dataset])
                check(tag + ' training clip', float(training.get('clip_value', -1.0)) == 1.0)
            results = cfg.get('results')
            check(tag + ' has results block', isinstance(results, dict))
            if not isinstance(results, dict):
                continue
            per_seed = results.get('per_seed', {})
            check(tag + ' has {} seeds'.format(EXPECTED_SEEDS_PER_GROUP),
                  len(per_seed) == EXPECTED_SEEDS_PER_GROUP, str(len(per_seed)))
            group = [r for r in rows if r['model'] == model and r['dataset'] == dataset]
            for r in group:
                seed = int(r['seed'].replace('seed', ''))
                entry = per_seed.get(seed)
                check(tag + ' seed{} entry'.format(seed), entry is not None)
                if entry is None:
                    continue
                check(tag + ' seed{} epoch'.format(seed), entry['epoch'] == int(r['best_epoch']))
                check(tag + ' seed{} ba'.format(seed), abs(entry['ba'] - float(r['test_ba'])) < 5e-5)
                check(tag + ' seed{} primary'.format(seed),
                      abs(entry[r['primary_name']] - float(r['test_primary'])) < 5e-5)
                check(tag + ' seed{} secondary'.format(seed),
                      abs(entry[r['secondary_name']] - float(r['test_secondary'])) < 5e-5)
            if group:
                for field, label in (('test_ba', 'ba'), ('test_primary', group[0]['primary_name']),
                                     ('test_secondary', group[0]['secondary_name'])):
                    values = [float(r[field]) for r in group]
                    check(tag + ' mean {}'.format(label),
                          abs(results[label]['mean'] - statistics.mean(values)) < 5e-5)
                    check(tag + ' std {}'.format(label),
                          abs(results[label]['std'] - statistics.pstdev(values)) < 5e-5)
            output = cfg.get('output') or {}
            roots = str(output.get('model_root', '')) + ' ' + str(output.get('log_root', ''))
            check(tag + ' output roots v2', 'foundation_rerun_v2_warm3_ema995_wd5e4_5seed_v1' in roots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=DEFAULT_CSV,
                        help='results CSV produced by collect_foundation_results.py')
    parser.add_argument('--yaml_root', default='configs/foundation_models',
                        help='root of the v2 YAML configs (override for dry-run tests)')
    args = parser.parse_args()

    yaml_root = Path(args.yaml_root)
    if not yaml_root.is_absolute():
        yaml_root = ROOT / yaml_root

    data_checks()
    rows = run_checks(args.csv)
    dataset_file_checks(rows, yaml_root)

    print('verification checks: {}/{} passed'.format(len(CHECKS) - len(FAILURES), len(CHECKS)))
    if FAILURES:
        print('FAILURES ({}):'.format(len(FAILURES)))
        for item in FAILURES[:80]:
            print('  FAIL:', item)
        raise SystemExit(1)
    print('all checks passed')


if __name__ == '__main__':
    main()
