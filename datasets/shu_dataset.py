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
            clip_limit=512.0,
            scale=32.0,
            bandpass_low=None,
            bandpass_high=None,
            filter_order=4,
            sampling_rate=200.0,
    ):
        super(CustomDataset, self).__init__()
        self.db = open_lmdb(data_dir)
        self.clip_limit = float(clip_limit)
        self.scale = float(scale)
        if self.clip_limit <= 0:
            raise ValueError('SHU-MI clip limit must be positive')
        self.bandpass_sos = None
        if bandpass_low is not None or bandpass_high is not None:
            if bandpass_low is None or bandpass_high is None:
                raise ValueError('SHU-MI band-pass requires both low and high cutoffs')
            low = float(bandpass_low)
            high = float(bandpass_high)
            order = int(filter_order)
            nyquist = float(sampling_rate) / 2.0
            if not 0 < low < high < nyquist:
                raise ValueError(
                    'SHU-MI band-pass must satisfy 0 < low < high < {} Hz'.format(nyquist)
                )
            if order <= 0:
                raise ValueError('SHU-MI filter order must be positive')
            self.bandpass_sos = signal.butter(
                order,
                [low, high],
                btype='bandpass',
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

        if self.bandpass_sos is not None:
            # LMDB records are [C, 4, 200].  Join them into the continuous
            # four-second trial before filtering; otherwise each one-second
            # reshape boundary would introduce an artificial edge transient.
            data = np.asarray(data, dtype=np.float32).reshape(data.shape[0], -1)
            data = signal.sosfiltfilt(self.bandpass_sos, data, axis=-1).astype(
                np.float32,
                copy=False,
            )
        data = clip_eeg(data, limit=self.clip_limit, scale=self.scale)
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
        clip_limit = getattr(self.params, 'shu_clip_limit', 512.0)
        scale = getattr(self.params, 'shu_scale', 32.0)
        bandpass_low = getattr(self.params, 'shu_bandpass_low', None)
        bandpass_high = getattr(self.params, 'shu_bandpass_high', None)
        filter_order = getattr(self.params, 'shu_filter_order', 4)
        dataset_kwargs = {
            'clip_limit': clip_limit,
            'scale': scale,
            'bandpass_low': bandpass_low,
            'bandpass_high': bandpass_high,
            'filter_order': filter_order,
        }
        if bandpass_low is not None or bandpass_high is not None:
            print(
                'SHU-MI zero-phase Butterworth band-pass enabled: {}-{} Hz, order={}'.format(
                    bandpass_low,
                    bandpass_high,
                    filter_order,
                )
            )
        train_set = CustomDataset(self.datasets_dir, mode='train', **dataset_kwargs)
        val_set = CustomDataset(self.datasets_dir, mode='val', **dataset_kwargs)
        test_set = CustomDataset(self.datasets_dir, mode='test', **dataset_kwargs)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': make_train_loader(train_set, self.params, train_set.collate),
            'val': make_eval_loader(val_set, self.params, val_set.collate),
            'test': make_eval_loader(test_set, self.params, test_set.collate),
        }
        return data_loader
