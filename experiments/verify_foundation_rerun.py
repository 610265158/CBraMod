#!/usr/bin/env python
"""End-to-end verification of the foundation-matched-pipeline rerun.

Part A - data pipeline: split counts, sample shapes/labels, and the numerical
equivalence loader(view) == raw / released_scale for all six model/dataset
combinations.

Part B - run records: for each of the 30 runs, cross-check the log command
line (model arch, dataset dir, lr, epochs, selection metric, single final
test), the printed normalization scale, the test metrics against the results
CSV, the selected-epoch checkpoint, and the YAML per_seed / mean /
std_population blocks against the CSV.

Exits non-zero when any check fails; prints only failures plus a summary.
"""
import csv
import glob
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
    ('reve', 'FACED'): '1000.0',
    ('reve', 'TUAB'): '100.0',
    ('reve', 'HMC'): '10.0',
}
EXPECTED_EPOCHS = {'FACED': '50', 'TUAB': '5', 'HMC': '30'}
EXPECTED_SELECTION = {'FACED': 'kappa', 'TUAB': 'pr_auc', 'HMC': 'kappa'}
EXPECTED_DIR = {'FACED': 'faced/processed', 'TUAB': 'TUAB', 'HMC': 'haaglanden'}
EXPECTED_COUNTS = {
    'FACED': {'train': 6720, 'val': 1680, 'test': 1932},
    'TUAB': {'train': 297103, 'val': 75407, 'test': 36945},
    'HMC': {'train': 91248, 'val': 22124, 'test': 23871},
}
BIG = ROOT.parent / 'BigDownstream'
HMC_DIR = BIG / 'haaglanden-medisch-centrum-sleep-staging-database-1.1' / 'processed'


def data_checks():
    from datasets import shape_utils as su

    import lmdb
    faced_dir = BIG / 'faced' / 'processed'
    env = lmdb.open(str(faced_dir), readonly=True, lock=False, readahead=False)
    with env.begin(write=False) as txn:
        faced_keys = pickle.loads(txn.get(b'__keys__'))
    env.close()
    counts = {key: len(value) for key, value in faced_keys.items()}
    check('FACED split counts', counts == EXPECTED_COUNTS['FACED'], str(counts))

    from datasets.faced_dataset import CustomDataset as FacedDS
    for scale in (100.0, 1000.0):
        su.configure_eeg_normalization(limit=float('inf'), scale=scale)
        x, y = FacedDS(str(faced_dir), mode='train')[0]
        env = lmdb.open(str(faced_dir), readonly=True, lock=False, readahead=False)
        with env.begin(write=False) as txn:
            raw = pickle.loads(txn.get(faced_keys['train'][0].encode()))['sample']
        env.close()
        expected = np.asarray(raw, dtype=np.float32) / np.float32(scale)
        check('FACED numeric raw/{}'.format(scale), np.allclose(x, expected))
        check('FACED shape (32,2000)', x.shape == (32, 2000), str(x.shape))
        check('FACED label 0-8', 0 <= int(y) <= 8, str(y))
        check('FACED finite', bool(np.isfinite(x).all()))

    tuab_dir = BIG / 'TUAB'
    for split, expected in EXPECTED_COUNTS['TUAB'].items():
        n = len(os.listdir(tuab_dir / split))
        check('TUAB {} count'.format(split), n == expected, str(n))
    from datasets.tuab_dataset import CustomDataset as TuabDS
    su.configure_eeg_normalization(limit=float('inf'), scale=100.0)
    x, y = TuabDS(str(tuab_dir), mode='train')[0]
    train_files = [str(tuab_dir / 'train' / name) for name in os.listdir(tuab_dir / 'train')]
    raw = pickle.load(open(train_files[0], 'rb'))['X']
    check('TUAB numeric raw/100', np.allclose(x, np.asarray(raw, dtype=np.float32) / np.float32(100.0)))
    check('TUAB shape (16,2000)', x.shape == (16, 2000), str(x.shape))
    check('TUAB label in {0,1}', int(y) in (0, 1), str(y))
    check('TUAB finite', bool(np.isfinite(x).all()))

    for split, expected in EXPECTED_COUNTS['HMC'].items():
        n = len([f for f in os.listdir(HMC_DIR / split) if f.endswith('.pkl')])
        check('HMC {} count'.format(split), n == expected, str(n))
    from datasets.hmc_dataset import CustomDataset as HmcDS
    for scale in (10.0, 100.0):
        su.configure_eeg_normalization(limit=float('inf'), scale=scale)
        x, y = HmcDS(str(HMC_DIR), mode='train')[0]
        first = sorted(glob.glob(str(HMC_DIR / 'train' / '*.pkl')))[0]
        raw = pickle.load(open(first, 'rb'))['X']
        check('HMC numeric raw/{}'.format(scale), np.allclose(x, np.asarray(raw, dtype=np.float32) / np.float32(scale)))
        check('HMC shape (4,6000)', x.shape == (4, 6000), str(x.shape))
        check('HMC label 0-4', 0 <= int(y) <= 4, str(y))
        check('HMC finite', bool(np.isfinite(x).all()))
    su.reset_eeg_normalization()


def run_checks():
    csv_path = ROOT / 'experiments/reports/foundation_rerun_results.csv'
    rows = [r for r in csv.DictReader(csv_path.open(encoding='utf-8')) if r['status'] == 'done']
    check('completed runs == 30', len(rows) == 30, str(len(rows)))

    queue_log = (ROOT / 'experiments/logs/foundation_rerun/queue_main.log').read_text(errors='replace')
    check('queue has no FAILED', 'FAILED' not in queue_log)
    done_sentinels = list((ROOT / 'experiments/checkpoints/foundation_rerun').rglob('.done'))
    check('30 .done sentinels', len(done_sentinels) == 30, str(len(done_sentinels)))

    for r in rows:
        model, dataset, seed = r['model'], r['dataset'], r['seed']
        tag = '{} {} {}'.format(model, dataset, seed)
        text = (ROOT / r['log']).read_text(errors='replace').replace('\r', '')

        check(tag + ' cmd model_arch', '--model_arch {}'.format(model) in text)
        check(tag + ' cmd datasets_dir', EXPECTED_DIR[dataset] in text)
        check(tag + ' cmd lr', '--lr 0.0001' in text)
        check(tag + ' cmd epochs', '--epochs {}'.format(EXPECTED_EPOCHS[dataset]) in text)
        check(tag + ' cmd selection', '--selection_metric {}'.format(EXPECTED_SELECTION[dataset]) in text)
        check(tag + ' cmd test_each_epoch false', '--test_each_epoch false' in text)
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

        ckpt_dir = (ROOT / r['checkpoint']).parent
        check(tag + ' best.pth exists', (ckpt_dir / 'best.pth').is_file())
        check(tag + ' selected-epoch checkpoint', len(list(ckpt_dir.glob('epoch{}_*.pth'.format(r['best_epoch'])))) >= 1)

    for model in ('cbramod', 'reve'):
        for dataset in ('FACED', 'TUAB', 'HMC'):
            cfg = yaml.safe_load((ROOT / 'configs/foundation_models' / model / '{}.yaml'.format(dataset)).read_text(encoding='utf-8'))
            res = cfg['results']
            group = [r for r in rows if r['model'] == model and r['dataset'] == dataset]
            check('yaml {}-{} has 5 seeds'.format(model, dataset), len(res['per_seed']) == 5)
            check('yaml {}-{} lr locked'.format(model, dataset), float(cfg['training']['lr']) == 0.0001)
            for r in group:
                seed = int(r['seed'].replace('seed', ''))
                entry = res['per_seed'][seed]
                check('yaml {}-{} seed{} epoch'.format(model, dataset, seed), entry['epoch'] == int(r['best_epoch']))
                check('yaml {}-{} seed{} ba'.format(model, dataset, seed), abs(entry['ba'] - float(r['test_ba'])) < 5e-5)
                check('yaml {}-{} seed{} primary'.format(model, dataset, seed),
                      abs(entry[r['primary_name']] - float(r['test_primary'])) < 5e-5)
                check('yaml {}-{} seed{} secondary'.format(model, dataset, seed),
                      abs(entry[r['secondary_name']] - float(r['test_secondary'])) < 5e-5)
            for field, label in (('test_ba', 'ba'), ('test_primary', group[0]['primary_name']),
                                 ('test_secondary', group[0]['secondary_name'])):
                values = [float(r[field]) for r in group]
                check('yaml {}-{} mean {}'.format(model, dataset, label),
                      abs(res[label]['mean'] - statistics.mean(values)) < 5e-5)
                check('yaml {}-{} std {}'.format(model, dataset, label),
                      abs(res[label]['std'] - statistics.pstdev(values)) < 5e-5)


def main():
    data_checks()
    run_checks()
    print('verification checks: {}/{} passed'.format(len(CHECKS) - len(FAILURES), len(CHECKS)))
    if FAILURES:
        print('FAILURES ({}):'.format(len(FAILURES)))
        for item in FAILURES[:60]:
            print('  FAIL:', item)
        raise SystemExit(1)
    print('all checks passed')


if __name__ == '__main__':
    main()
