import pickle
import os
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

root = os.environ.get("CHB_CLEAN_PATH", "/data/datasets/BigDownstream/chb-mit/processed")
out = os.environ.get("CHB_SEG_PATH", "/data/datasets/BigDownstream/chb-mit/processed_seg")

# dump chb23 and chb24 to test, ch21 and ch22 to val, and the rest to train
test_pats = ["chb23", "chb24"]
val_pats = ["chb21", "chb22"]
train_pats = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
    "chb06",
    "chb07",
    "chb08",
    "chb09",
    "chb10",
    "chb11",
    "chb12",
    "chb13",
    "chb14",
    "chb15",
    "chb16",
    "chb17",
    "chb18",
    "chb19",
    "chb20",
]
channels = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]
SAMPLING_RATE = 256
SEGMENT_POINTS = SAMPLING_RATE * 10
SEIZURE_AUG_STEP = SAMPLING_RATE * 5


def segment_path(out_folder, filename):
    return os.path.join(out_folder, filename)


def expected_segment_filenames(record_file, signal_points, seizure_times):
    stem = record_file.split(".")[0]
    filenames = []
    for i in range(0, signal_points, SEGMENT_POINTS):
        if i + SEGMENT_POINTS <= signal_points:
            filenames.append(f"{stem}-{i}.pkl")

    for idx, seizure_time in enumerate(seizure_times):
        for i in range(
            max(0, seizure_time[0] - SAMPLING_RATE),
            min(seizure_time[1] + SAMPLING_RATE, signal_points),
            SEIZURE_AUG_STEP,
        ):
            filenames.append(f"{stem}-s-{idx}-add-{i}.pkl")
    return filenames


def outputs_complete(out_folder, record_file, signal_points, seizure_times):
    return all(
        os.path.exists(segment_path(out_folder, filename))
        for filename in expected_segment_filenames(record_file, signal_points, seizure_times)
    )


def dump_segment(out_path, segment, label, skip_existing):
    if skip_existing and os.path.exists(out_path):
        return
    pickle.dump({"X": segment, "y": label}, open(out_path, "wb"))


def sub_to_segments(folder, out_folder, skip_existing=True):
    print(f"Processing {folder}...")
    # each recording
    for f in tqdm(os.listdir(os.path.join(root, folder))):
        print(f"Processing {folder}/{f}...")
        record = pickle.load(open(os.path.join(root, folder, f), "rb"))
        """
        {'FP1-F7': array([-145.93406593,    0.1953602 ,    0.1953602 , ...,  -11.52625153, -2.93040293,   19.34065934]), 
         'F7-T7': array([-104.51770452,    0.1953602 ,    0.1953602 , ...,   23.63858364, 27.54578755,   30.67155067]), 
         'T7-P7': array([-42.78388278,   0.1953602 ,   0.1953602 , ...,  48.64468864, 45.12820513,  34.57875458]), 
        'P7-O1': array([-33.01587302,   0.1953602 ,   0.1953602 , ..., -17.77777778, -20.51282051, -25.59218559]), 
       'FP1-F3': array([-170.94017094,    0.1953602 ,    0.1953602 , ...,  -34.96947497, -25.98290598,    0.1953602 ]), 
        'F3-C3': array([-110.76923077,    0.1953602 ,    0.1953602 , ...,   38.0952381 , 48.64468864,   50.20757021]), 
         'C3-P3': array([11.91697192,  0.1953602 ,  0.1953602 , ..., 40.04884005, 33.7973138 , 25.98290598]), 
       'P3-O1': array([-56.45909646,   0.1953602 ,   0.1953602 , ...,   0.97680098, -6.44688645, -16.60561661]), 
        'FP2-F4': array([-139.29181929,    0.1953602 ,    0.1953602 , ...,   -2.14896215, -2.14896215,   -0.58608059]), 
         'F4-C4': array([-1.36752137,  0.1953602 ,  0.1953602 , ...,  1.75824176, 2.93040293,  7.22832723]), 
        'C4-P4': array([63.88278388,  0.1953602 ,  0.1953602 , ..., 16.996337  , 23.63858364, 25.59218559]), 
       'P4-O2': array([-14.26129426,   0.1953602 ,   0.1953602 , ..., -13.08913309, -8.00976801, -13.47985348]), 
        'FP2-F8': array([-2.67838828e+02,  1.95360195e-01,  1.95360195e-01, ..., 6.83760684e+00,  6.05616606e+00,  6.44688645e+00]), 
        'F8-T8': array([ 57.24053724,   0.1953602 ,   0.1953602 , ...,  -2.53968254,  -9.96336996, -12.6984127 ]), 
        'T8-P8': array([44.73748474,  0.1953602 ,  0.1953602 , ..., 16.996337  , 22.46642247, 26.37362637]), 
       'P8-O2': array([ 74.82295482,   0.1953602 ,  -0.1953602 , ..., -17.38705739, -1.75824176,  -2.53968254]), 
        'FZ-CZ': array([-106.08058608,    0.1953602 ,    0.1953602 , ...,   24.81074481, 28.71794872,   28.71794872]), 
         'CZ-PZ': array([84.59096459,  0.1953602 ,  0.1953602 , ..., 18.94993895, 20.51282051, 18.16849817]), 
       'P7-T7': array([ 43.17460317,   0.1953602 ,   0.1953602 , ..., -48.25396825, -44.73748474, -34.18803419]), 
       'T7-FT9': array([-57.24053724,   0.1953602 ,   0.1953602 , ..., -11.91697192,  -3.71184371,   2.14896215]), 
        'FT9-FT10': array([-2.64713065e+02,  1.95360195e-01,  5.86080586e-01, ..., 9.76800977e-01, -1.58241758e+01, -2.94993895e+01]), 
        'FT10-T8': array([ 94.74969475,   0.1953602 ,   0.1953602 , ...,  -7.22832723, -10.35409035, -13.47985348]), 
       'T8-P8-2': array([44.73748474,  0.1953602 ,  0.1953602 , ..., 16.996337  , 22.46642247, 26.37362637]), 
       'metadata': {'seizures': 0, 'times': [], 'channels': ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-2']}}
        """
        if "times" in record["metadata"]:
            seizure_times = record["metadata"]["times"]
        else:
            seizure_times = []

        missing_channels = [channel for channel in channels if channel not in record]
        if missing_channels:
            raise ValueError(f"Channels {missing_channels} not found in record {f}")

        signal_points = len(record[channels[0]])
        if skip_existing and outputs_complete(out_folder, f, signal_points, seizure_times):
            print(f"Skipping complete recording {folder}/{f}")
            continue

        signal = np.array([record[channel] for channel in channels])

        # split the signal into segments on the second dimension by SAMPLING_RATE * 10 seconds
        for i in range(0, signal.shape[1], SEGMENT_POINTS):
            segment = signal[:, i : i + SEGMENT_POINTS]
            if segment.shape[1] == SEGMENT_POINTS:
                # judge whether the segment contains seizures
                label = 0

                for seizure_time in seizure_times:
                    if (
                        i < seizure_time[0] < i + SEGMENT_POINTS
                        or i < seizure_time[1] < i + SEGMENT_POINTS
                    ):
                        label = 1
                        break

                # save the segment
                dump_segment(
                    os.path.join(out_folder, f"{f.split('.')[0]}-{i}.pkl"),
                    segment,
                    label,
                    skip_existing,
                )

        for idx, seizure_time in enumerate(seizure_times):
            for i in range(
                max(0, seizure_time[0] - SAMPLING_RATE),
                min(seizure_time[1] + SAMPLING_RATE, signal.shape[1]),
                SEIZURE_AUG_STEP,
            ):
                segment = signal[:, i : i + SEGMENT_POINTS]
                label = 1
                # save the segment
                dump_segment(
                    os.path.join(
                        out_folder, f"{f.split('.')[0]}-s-{idx}-add-{i}.pkl"
                    ),
                    segment,
                    label,
                    skip_existing,
                )


def main():
    global root, out
    parser = argparse.ArgumentParser(description="Segment cleaned CHB-MIT pkl files.")
    parser.add_argument("--root", default=root, help="Input cleaned CHB-MIT pkl directory.")
    parser.add_argument("--out", default=out, help="Output segmented CHB-MIT directory.")
    parser.add_argument(
        "--processes",
        type=int,
        default=int(os.environ.get("CHB_SEG_PROCESSES", max(1, mp.cpu_count() // 2))),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Overwrite existing output files instead of resuming partial runs.",
    )
    args = parser.parse_args()

    root = args.root
    out = args.out
    os.makedirs(out, exist_ok=True)

    folders = os.listdir(root)
    out_folders = []
    for folder in folders:
        if folder in test_pats:
            out_folder = os.path.join(out, "test")
        elif folder in val_pats:
            out_folder = os.path.join(out, "val")
        else:
            out_folder = os.path.join(out, "train")

        os.makedirs(out_folder, exist_ok=True)
        out_folders.append(out_folder)

    with mp.Pool(args.processes) as pool:
        pool.starmap(
            sub_to_segments,
            [(folder, out_folder, not args.no_skip_existing) for folder, out_folder in zip(folders, out_folders)],
        )


if __name__ == "__main__":
    main()
