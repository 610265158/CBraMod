"""Preprocess the Haaglanden Medisch Centrum sleep staging database.

The HMC release contains one signal EDF and one ``*_sleepscoring.txt`` file
per subject.  This script follows the NeuroLM preprocessing recipe: retain the
four standard EEG channels, band-pass 0.1--75 Hz, notch 50 Hz, resample to
200 Hz, and write one 30-second epoch per pickle file.

Example::

    python preprocessing/preprocessing_hmc.py \
      --raw /nas/public/haaglanden-medisch-centrum-sleep-staging-database-1.1/recordings \
      --processed /data/lz/public/BigDownstream/haaglanden-medisch-centrum-sleep-staging-database-1.1/processed
"""

import argparse
import os
import pickle

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm


SIGNAL_CHANNELS = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2"]
DROP_CHANNELS = ["EMG chin", "EOG E1-M2", "EOG E2-M2", "ECG"]
LABELS = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}
TARGET_FS = 200
EPOCH_SECONDS = 30


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Directory containing signal EDFs")
    parser.add_argument("--processed", required=True, help="Output directory")
    parser.add_argument("--jobs", type=int, default=None, help="MNE filter/resample workers")
    return parser.parse_args()


def signal_files(raw_dir):
    files = []
    for name in os.listdir(raw_dir):
        if not (name.endswith(".edf") and not name.endswith("_sleepscoring.edf")):
            continue
        path = os.path.join(raw_dir, name)
        if os.path.isfile(path) and os.path.exists(path[:-4] + "_sleepscoring.txt"):
            files.append(path)
    return sorted(files)


def read_epochs(edf_path, jobs):
    scoring_path = edf_path[:-4] + "_sleepscoring.txt"
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    missing = [channel for channel in SIGNAL_CHANNELS if channel not in raw.ch_names]
    if missing:
        raise ValueError(f"{os.path.basename(edf_path)} missing channels: {missing}")
    raw.pick(SIGNAL_CHANNELS)
    raw.reorder_channels(SIGNAL_CHANNELS)
    raw.filter(l_freq=0.1, h_freq=75.0, n_jobs=jobs, verbose=False)
    raw.notch_filter(50.0, n_jobs=jobs, verbose=False)
    raw.resample(TARGET_FS, n_jobs=jobs, verbose=False)
    signals = raw.get_data(units="uV").astype(np.float32, copy=False)
    raw.close()

    annotations = pd.read_csv(scoring_path, skipinitialspace=True)
    annotations.columns = [str(column).strip() for column in annotations.columns]
    required = {"Recording onset", "Duration", "Annotation"}
    missing_columns = required.difference(annotations.columns)
    if missing_columns:
        raise ValueError(f"{scoring_path} missing columns: {sorted(missing_columns)}")

    samples_per_epoch = TARGET_FS * EPOCH_SECONDS
    epochs = []
    skipped = 0
    for _, row in annotations.iterrows():
        duration = float(row["Duration"])
        annotation = str(row["Annotation"]).strip()
        if duration != EPOCH_SECONDS or annotation not in LABELS:
            skipped += 1
            continue
        onset = float(row["Recording onset"])
        start = int(round(onset * TARGET_FS))
        end = start + samples_per_epoch
        if start < 0 or end > signals.shape[1]:
            skipped += 1
            continue
        epochs.append((signals[:, start:end], LABELS[annotation]))
    if not epochs:
        raise ValueError(f"No valid 30-second epochs found for {edf_path}")
    return epochs, skipped


def write_subject(edf_path, out_dir, jobs):
    epochs, skipped = read_epochs(edf_path, jobs)
    stem = os.path.splitext(os.path.basename(edf_path))[0]
    for index, (signal, label) in enumerate(tqdm(epochs, desc=stem, leave=False)):
        if signal.shape != (len(SIGNAL_CHANNELS), TARGET_FS * EPOCH_SECONDS):
            raise ValueError(f"Unexpected epoch shape {signal.shape} in {edf_path}")
        sample = {
            "X": signal,
            "ch_names": ["F4", "C4", "O2", "C3"],
            "y": int(label),
        }
        with open(os.path.join(out_dir, f"{stem}-{index}.pkl"), "wb") as handle:
            pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return len(epochs), skipped


def main():
    args = parse_args()
    jobs = args.jobs if args.jobs is not None else max(1, min(16, (os.cpu_count() or 2) - 1))
    files = signal_files(args.raw)
    if len(files) < 3:
        raise SystemExit(f"Expected at least 3 signal EDFs with scoring text, found {len(files)}")

    split_names = (("train", files[:100]), ("val", files[100:125]), ("test", files[125:]))
    totals = {"subjects": 0, "epochs": 0, "skipped": 0}
    for split, split_files in split_names:
        split_dir = os.path.join(args.processed, split)
        os.makedirs(split_dir, exist_ok=True)
        for edf_path in split_files:
            print(f"Processing {edf_path}")
            try:
                epochs, skipped = write_subject(edf_path, split_dir, jobs)
            except (OSError, ValueError, KeyError) as exc:
                raise RuntimeError(f"Failed to preprocess {edf_path}: {exc}") from exc
            totals["subjects"] += 1
            totals["epochs"] += epochs
            totals["skipped"] += skipped
            print(f"  wrote {epochs} epochs; skipped {skipped} annotation rows")
    print(
        "Finished HMC preprocessing: "
        f"{totals['subjects']} subjects, {totals['epochs']} epochs, "
        f"{totals['skipped']} skipped rows"
    )


if __name__ == "__main__":
    main()
