import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import pickle
from scipy import signal

from datasets.sampling import make_eval_loader, make_train_loader
from datasets.shape_utils import clip_eeg


CORRUPT_FILES = {
    'chb04_19-3563520.pkl',
    'chb06_03-3571200.pkl',
    'chb09_15-1973760.pkl',
    'chb22_30-87040.pkl',
    'chb23_16-2457600.pkl',
}


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        mode_dir = os.path.join(data_dir, mode)
        self.files = [
            os.path.join(mode_dir, file)
            for file in os.listdir(mode_dir)
            if file not in CORRUPT_FILES
        ]


    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        try:
            with open(file, 'rb') as f:
                data_dict = pickle.load(f)
        except Exception as exc:
            raise RuntimeError(f'Failed to load CHB-MIT sample: {file}') from exc
        data = data_dict['X']
        label = data_dict['y']
        data = signal.resample(data, 2000, axis=1)
        data = clip_eeg(data)
        return data, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        data_loader = {
            'train': make_train_loader(train_set, self.params, train_set.collate),
            'val': make_eval_loader(val_set, self.params, val_set.collate),
            'test': make_eval_loader(test_set, self.params, test_set.collate),
        }
        return data_loader
