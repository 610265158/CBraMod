import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
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
    ):
        super(CustomDataset, self).__init__()
        self.db = open_lmdb(data_dir)
        self.clip_limit = float(clip_limit)
        self.scale = float(scale)
        if self.clip_limit <= 0:
            raise ValueError('SHU-MI clip limit must be positive')
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
        train_set = CustomDataset(self.datasets_dir, mode='train', clip_limit=clip_limit, scale=scale)
        val_set = CustomDataset(self.datasets_dir, mode='val', clip_limit=clip_limit, scale=scale)
        test_set = CustomDataset(self.datasets_dir, mode='test', clip_limit=clip_limit, scale=scale)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': make_train_loader(train_set, self.params, train_set.collate),
            'val': make_eval_loader(val_set, self.params, val_set.collate),
            'test': make_eval_loader(test_set, self.params, test_set.collate),
        }
        return data_loader
