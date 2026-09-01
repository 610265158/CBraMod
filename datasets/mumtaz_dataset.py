import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
from utils.util import to_tensor
import os
import random
import pickle
from datasets.lmdb_utils import open_lmdb
from datasets.sampling import make_eval_loader, make_train_loader
from datasets.shape_utils import as_channel_time, clip_eeg


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            lowpass_hz=None,
            filter_order=4,
            sampling_rate=200.0,
    ):
        super(CustomDataset, self).__init__()
        self.db = open_lmdb(data_dir)
        self.lowpass_sos = None
        if lowpass_hz is not None:
            cutoff = float(lowpass_hz)
            order = int(filter_order)
            nyquist = float(sampling_rate) / 2.0
            if not 0 < cutoff < nyquist:
                raise ValueError(
                    'Mumtaz2016 low-pass must satisfy 0 < cutoff < {} Hz'.format(nyquist)
                )
            if order <= 0:
                raise ValueError('Mumtaz2016 filter order must be positive')
            self.lowpass_sos = signal.butter(
                order,
                cutoff,
                btype='lowpass',
                fs=float(sampling_rate),
                output='sos',
            )
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        if self.lowpass_sos is not None:
            # LMDB samples are stored as [C, 5, 200]. Join the five seconds
            # before filtering so internal storage boundaries create no edge.
            data = np.asarray(data, dtype=np.float32).reshape(data.shape[0], -1)
            data = signal.sosfiltfilt(self.lowpass_sos, data, axis=-1).astype(
                np.float32,
                copy=False,
            )
        data = clip_eeg(data)
        return data, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(as_channel_time(x_data)), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        lowpass_hz = getattr(self.params, 'mumtaz_lowpass_hz', None)
        filter_order = getattr(self.params, 'mumtaz_filter_order', 4)
        dataset_kwargs = {
            'lowpass_hz': lowpass_hz,
            'filter_order': filter_order,
        }
        if lowpass_hz is not None:
            print(
                'Mumtaz2016 zero-phase Butterworth low-pass enabled: {} Hz, order={}'.format(
                    lowpass_hz,
                    filter_order,
                )
            )
        train_set = CustomDataset(self.datasets_dir, mode='train', **dataset_kwargs)
        val_set = CustomDataset(self.datasets_dir, mode='val', **dataset_kwargs)
        test_set = CustomDataset(self.datasets_dir, mode='test', **dataset_kwargs)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        data_loader = {
            'train': make_train_loader(train_set, self.params, train_set.collate),
            'val': make_eval_loader(val_set, self.params, val_set.collate),
            'test': make_eval_loader(test_set, self.params, test_set.collate),
        }
        return data_loader
