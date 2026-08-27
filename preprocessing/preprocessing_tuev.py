import mne
import numpy as np
import os
import pickle
import shutil
from scipy import signal as scipy_signal
from tqdm import tqdm

if not hasattr(np, "in1d"):
    np.in1d = np.isin

"""
https://github.com/Abhishaike/EEG_Event_Classification
"""


def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
    fs = 200.0
    [numChan, numPoints] = signals.shape
    # for i in range(numChan):  # standardize each channel
    #     if np.std(signals[i, :]) > 0:
    #         signals[i, :] = (signals[i, :] - np.mean(signals[i, :])) / np.std(signals[i, :])
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])  # channel that had the detected thing
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):  # for each event
        chan = int(EventData[i, 0])  # chan is channel
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        # print (offset + start - 2 * int(fs), offset + end + 2 * int(fs), signals.shape)
        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels]


def convert_signals(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]]
            - signals[signal_names["EEG F7-REF"]],  # 0
            (
                signals[signal_names["EEG F7-REF"]]
                - signals[signal_names["EEG T3-REF"]]
            ),  # 1
            (
                signals[signal_names["EEG T3-REF"]]
                - signals[signal_names["EEG T5-REF"]]
            ),  # 2
            (
                signals[signal_names["EEG T5-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 3
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F8-REF"]]
            ),  # 4
            (
                signals[signal_names["EEG F8-REF"]]
                - signals[signal_names["EEG T4-REF"]]
            ),  # 5
            (
                signals[signal_names["EEG T4-REF"]]
                - signals[signal_names["EEG T6-REF"]]
            ),  # 6
            (
                signals[signal_names["EEG T6-REF"]]
                - signals[signal_names["EEG O2-REF"]]
            ),  # 7
            (
                signals[signal_names["EEG FP1-REF"]]
                - signals[signal_names["EEG F3-REF"]]
            ),  # 14
            (
                signals[signal_names["EEG F3-REF"]]
                - signals[signal_names["EEG C3-REF"]]
            ),  # 15
            (
                signals[signal_names["EEG C3-REF"]]
                - signals[signal_names["EEG P3-REF"]]
            ),  # 16
            (
                signals[signal_names["EEG P3-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 17
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F4-REF"]]
            ),  # 18
            (
                signals[signal_names["EEG F4-REF"]]
                - signals[signal_names["EEG C4-REF"]]
            ),  # 19
            (
                signals[signal_names["EEG C4-REF"]]
                - signals[signal_names["EEG P4-REF"]]
            ),  # 20
            (signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]]),
        )
    )  # 21
    return new_signals


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    sfreq = Rawdata.info["sfreq"]
    signals = Rawdata.get_data(units='uV')
    if sfreq != 200:
        target_points = int(round(signals.shape[1] * 200 / sfreq))
        signals = scipy_signal.resample(signals, target_points, axis=1)

    sos = scipy_signal.butter(5, [0.3, 75], btype="band", fs=200, output="sos")
    try:
        signals = scipy_signal.sosfiltfilt(sos, signals, axis=1)
    except ValueError:
        signals = scipy_signal.sosfilt(sos, signals, axis=1)

    b_notch, a_notch = scipy_signal.iirnotch(60, 30, fs=200)
    try:
        signals = scipy_signal.filtfilt(b_notch, a_notch, signals, axis=1)
    except ValueError:
        signals = scipy_signal.lfilter(b_notch, a_notch, signals, axis=1)

    times = np.arange(signals.shape[1]) / 200
    RecFile = fileName[0:-3] + "rec"
    eventData = read_event_data(RecFile)
    Rawdata.close()
    return [signals, times, eventData, Rawdata]


def read_event_data(rec_file):
    eventData = np.genfromtxt(rec_file, delimiter=",")
    return np.atleast_2d(eventData)


def output_complete(out_dir, fname, num_events):
    base = fname.split(".")[0]
    return all(
        os.path.exists(os.path.join(out_dir, base + "-" + str(idx) + ".pkl"))
        for idx in range(num_events)
    )


def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir, skip_existing=True):
    for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname[-4:] == ".edf":
                print("\t%s" % fname)
                RecFile = os.path.join(dirName, fname[0:-3] + "rec")
                try:
                    event = read_event_data(RecFile)
                except (ValueError, OSError):
                    print("could not read events in " + RecFile)
                    continue
                if skip_existing and output_complete(OutDir, fname, event.shape[0]):
                    print("\tskipping complete file")
                    continue
                try:
                    [signals, times, event, Rawdata] = readEDF(
                        dirName + "/" + fname
                    )  # event is the .rec file in the form of an array
                    signals = convert_signals(signals, Rawdata)
                except (ValueError, KeyError):
                    print("something funky happened in " + dirName + "/" + fname)
                    continue
                signals, offending_channels, labels = BuildEvents(signals, times, event)

                for idx, (signal, offending_channel, label) in enumerate(
                    zip(signals, offending_channels, labels)
                ):
                    sample = {
                        "signal": signal,
                        "offending_channel": offending_channel,
                        "label": label,
                    }
                    save_pickle(
                        sample,
                        os.path.join(
                            OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"
                        ),
                    )

    return Features, Labels, OffendingChannels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


"""
TUEV dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
"""

def env_flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no")


def parse_splits(value):
    return {split.strip() for split in value.split(",") if split.strip()}


def copy_if_needed(src, dst):
    if os.path.exists(dst) and os.path.getsize(src) == os.path.getsize(dst):
        return
    shutil.copy2(src, dst)


def finalize_splits(target):
    train_files = os.listdir(os.path.join(target, "processed_train"))
    train_val_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train val sub:", train_val_sub)
    test_files = os.listdir(os.path.join(target, "processed_eval"))

    train_val_sub.sort(key=lambda x: x)

    train_sub = train_val_sub[: int(len(train_val_sub) * 0.8)]
    val_sub = train_val_sub[int(len(train_val_sub) * 0.8) :]
    print("train sub:", train_sub)
    print("val sub:", val_sub)

    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    for split in ("processed_train", "processed_eval", "processed_test"):
        os.makedirs(os.path.join(target, "processed", split), exist_ok=True)

    for file in tqdm(train_files):
        copy_if_needed(
            os.path.join(target, "processed_train", file),
            os.path.join(target, "processed", "processed_train", file),
        )
    for file in tqdm(val_files):
        copy_if_needed(
            os.path.join(target, "processed_train", file),
            os.path.join(target, "processed", "processed_eval", file),
        )
    for file in tqdm(test_files):
        copy_if_needed(
            os.path.join(target, "processed_eval", file),
            os.path.join(target, "processed", "processed_test", file),
        )


def main():
    root = os.environ.get("TUEV_ROOT", "/data/zcb/data/TUEV/edf")
    target = os.environ.get("TUEV_TARGET", "/data/datasets/BigDownstream/TUEV_refine")
    process_splits = parse_splits(os.environ.get("TUEV_PROCESS_SPLITS", "train,eval"))
    skip_existing = env_flag("TUEV_SKIP_EXISTING", True)

    mne.set_log_level(os.environ.get("MNE_LOG_LEVEL", "WARNING"))

    train_out_dir = os.path.join(target, "processed_train")
    eval_out_dir = os.path.join(target, "processed_eval")

    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(eval_out_dir, exist_ok=True)

    fs = 200

    if "train" in process_splits:
        BaseDirTrain = os.path.join(root, "train")
        TrainFeatures = np.empty(
            (0, 16, fs)
        )  # 0 for lack of intialization, 22 for channels, fs for num of points
        TrainLabels = np.empty([0, 1])
        TrainOffendingChannel = np.empty([0, 1])
        load_up_objects(
            BaseDirTrain,
            TrainFeatures,
            TrainLabels,
            TrainOffendingChannel,
            train_out_dir,
            skip_existing=skip_existing,
        )

    if "eval" in process_splits:
        BaseDirEval = os.path.join(root, "eval")
        EvalFeatures = np.empty(
            (0, 16, fs)
        )  # 0 for lack of intialization, 22 for channels, fs for num of points
        EvalLabels = np.empty([0, 1])
        EvalOffendingChannel = np.empty([0, 1])
        load_up_objects(
            BaseDirEval,
            EvalFeatures,
            EvalLabels,
            EvalOffendingChannel,
            eval_out_dir,
            skip_existing=skip_existing,
        )

    if env_flag("TUEV_FINALIZE", True):
        finalize_splits(target)

    print("Done!")


if __name__ == "__main__":
    main()
