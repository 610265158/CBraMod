"""Build 16-channel raw-wave EEG-Vision LMDBs from TUSZ or TUAB EDFs."""

import argparse
import hashlib
import json
import pickle
import random
import re
from pathlib import Path

import lmdb
import mne
import numpy as np
from tqdm import tqdm


ELECTRODES = ('FP1', 'F7', 'T3', 'T5', 'O1', 'FP2', 'F8', 'T4', 'T6', 'O2', 'F3', 'C3', 'P3', 'F4', 'C4', 'P4')
MODERN_TO_LEGACY = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
BIPOLAR_PAIRS = (
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
)


def normalize_channel(name):
    name = re.sub(r'^EEG[\s_-]*', '', name.upper().strip())
    name = re.sub(r'[\s_-]*(REF|LE|AVG|AR)$', '', name)
    name = name.replace(' ', '').replace('-', '').replace('_', '')
    return MODERN_TO_LEGACY.get(name, name)


def electrode_names(raw):
    names = {}
    for raw_name in raw.ch_names:
        normalized = normalize_channel(raw_name)
        if normalized in ELECTRODES and normalized not in names:
            names[normalized] = raw_name
    missing = [name for name in ELECTRODES if name not in names]
    if missing:
        raise ValueError(f'missing electrodes: {", ".join(missing)}')
    return names


def discover_edfs(input_root, source, allow_eval):
    paths = sorted(path for path in input_root.rglob('*') if path.is_file() and path.suffix.lower() == '.edf')
    if source != 'tuab' or allow_eval:
        return paths, 0
    safe = []
    excluded = 0
    for path in paths:
        parts = {part.lower() for part in path.parts}
        if {'eval', 'evaluation', 'test'} & parts:
            excluded += 1
        else:
            safe.append(path)
    return safe, excluded


def recording_windows(path, args):
    raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    names = electrode_names(raw)
    picks = list(dict.fromkeys(names.values()))
    nyquist = raw.info['sfreq'] / 2
    raw.filter(args.low_freq, min(args.high_freq, nyquist * 0.95), picks=picks, method='iir', verbose='ERROR')
    if 0 < args.notch_freq < nyquist:
        raw.notch_filter(args.notch_freq, picks=picks, method='iir', verbose='ERROR')
    if raw.info['sfreq'] != args.sample_rate:
        raw.resample(args.sample_rate, npad='auto', verbose='ERROR')
    referential = raw.get_data(picks=[names[name] for name in ELECTRODES]).astype(np.float32) * 1e6
    index = {name: idx for idx, name in enumerate(ELECTRODES)}
    bipolar = np.stack(
        [referential[index[left]] - referential[index[right]] for left, right in BIPOLAR_PAIRS],
        axis=0,
    )
    trim = int(round(args.trim_seconds * args.sample_rate))
    if trim:
        if bipolar.shape[1] <= 2 * trim:
            raise ValueError('recording is too short after trimming')
        bipolar = bipolar[:, trim:-trim]
    points = args.window_seconds * args.sample_rate
    count = bipolar.shape[1] // points
    if count == 0:
        raise ValueError('recording contains no complete windows')
    return bipolar[:, :count * points].reshape(16, count, points).transpose(1, 0, 2).copy()


def parse_args():
    parser = argparse.ArgumentParser(description='Create EEG-Vision TUSZ/TUAB pretraining LMDB')
    parser.add_argument('--source', required=True, choices=['tusz', 'tuab'])
    parser.add_argument('--input_root', type=Path, required=True)
    parser.add_argument('--output_lmdb', type=Path, required=True)
    parser.add_argument('--sample_rate', type=int, default=200)
    parser.add_argument('--window_seconds', type=int, default=10)
    parser.add_argument('--trim_seconds', type=int, default=0)
    parser.add_argument('--low_freq', type=float, default=0.3)
    parser.add_argument('--high_freq', type=float, default=75.0)
    parser.add_argument('--notch_freq', type=float, default=60.0)
    parser.add_argument('--artifact_threshold_uv', type=float, default=0.0,
                        help='reject windows above this absolute peak; <=0 disables')
    parser.add_argument('--map_size_gb', type=float, default=512.0)
    parser.add_argument('--commit_interval', type=int, default=256)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--max_recordings', type=int, default=None)
    parser.add_argument('--allow_eval', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)
    if args.sample_rate != 200 or args.window_seconds != 10:
        raise ValueError('The current TUAB vision model expects 200 Hz, 10-second windows')
    if args.output_lmdb.exists() and not args.dry_run:
        raise FileExistsError(f'Refusing to overwrite {args.output_lmdb}')
    paths, excluded = discover_edfs(args.input_root, args.source, args.allow_eval)
    random.Random(args.seed).shuffle(paths)
    if args.max_recordings is not None:
        paths = paths[:args.max_recordings]
    if not paths:
        raise RuntimeError(f'No eligible EDFs found under {args.input_root}')
    print(f'Found {len(paths):,} EDFs; excluded {excluded:,} TUAB eval/test EDFs')

    db = txn = None
    if not args.dry_run:
        args.output_lmdb.parent.mkdir(parents=True, exist_ok=True)
        db = lmdb.open(str(args.output_lmdb), map_size=int(args.map_size_gb * 1024 ** 3))
        txn = db.begin(write=True)
    keys = []
    stats = {'source': args.source, 'recordings': len(paths), 'processed': 0, 'skipped': 0,
             'windows': 0, 'artifact_rejected': 0, 'excluded_eval': excluded,
             'shape': [16, 2000], 'unit': 'microvolts'}
    pending = 0
    try:
        for path in tqdm(paths):
            try:
                windows = recording_windows(path, args)
            except Exception as exc:
                stats['skipped'] += 1
                print(f'Skip {path}: {exc}')
                continue
            stats['processed'] += 1
            relative = path.relative_to(args.input_root).as_posix()
            digest = hashlib.sha1(relative.encode()).hexdigest()[:12]
            for idx, sample in enumerate(windows):
                if not np.isfinite(sample).all():
                    continue
                if args.artifact_threshold_uv > 0 and np.max(np.abs(sample)) >= args.artifact_threshold_uv:
                    stats['artifact_rejected'] += 1
                    continue
                key = f'{args.source}:{digest}:{path.stem}:{idx:06d}'
                keys.append(key)
                if txn is not None:
                    txn.put(key.encode(), pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
                    pending += 1
                    if pending >= args.commit_interval:
                        txn.commit()
                        txn = db.begin(write=True)
                        pending = 0
            if args.dry_run:
                print(f'{path}: {len(windows)} windows, shape={windows.shape[1:]}, range=[{windows.min():.1f}, {windows.max():.1f}] uV')
        stats['windows'] = len(keys)
        if txn is not None:
            txn.put(b'__keys__', pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL))
            txn.put(b'__meta__', json.dumps(stats, sort_keys=True).encode())
            txn.commit()
            txn = None
            db.sync()
    finally:
        if txn is not None:
            txn.abort()
        if db is not None:
            db.close()
    print(json.dumps(stats, indent=2, sort_keys=True))
    if not keys:
        raise RuntimeError('No windows were produced')


if __name__ == '__main__':
    main()
