import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import pickle
from datasets.lmdb_utils import open_lmdb
from datasets.sampling import make_eval_loader, make_train_loader
from datasets.shape_utils import clip_eeg, robust_scale_eeg
class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            input_norm='clip_scale',
            robust_clip=8.0,
    ):
        super(CustomDataset, self).__init__()
        self.db = open_lmdb(data_dir)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
        self.input_norm = input_norm
        self.robust_clip = float(robust_clip)
        if self.input_norm not in {'clip_scale', 'robust_sample'}:
            raise ValueError('Unsupported FACED input normalization: {}'.format(self.input_norm))

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        if self.input_norm == 'robust_sample':
            data = robust_scale_eeg(data, clip=self.robust_clip)
        else:
            data = clip_eeg(data)
        return data, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        input_norm = getattr(self.params, 'faced_input_norm', 'clip_scale')
        robust_clip = getattr(self.params, 'faced_robust_clip', 8.0)
        train_set = CustomDataset(self.datasets_dir, mode='train', input_norm=input_norm,
                                  robust_clip=robust_clip)
        val_set = CustomDataset(self.datasets_dir, mode='val', input_norm=input_norm,
                                robust_clip=robust_clip)
        test_set = CustomDataset(self.datasets_dir, mode='test', input_norm=input_norm,
                                 robust_clip=robust_clip)
        print('FACED input normalization: {}, robust clip: {}'.format(input_norm, robust_clip))
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': make_train_loader(train_set, self.params, train_set.collate),
            'val': make_eval_loader(val_set, self.params, val_set.collate),
            'test': make_eval_loader(test_set, self.params, test_set.collate),
        }
        return data_loader
