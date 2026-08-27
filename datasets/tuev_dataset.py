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


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            files,
    ):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.files = files

    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(os.path.join(self.data_dir, file), "rb"))
        data = data_dict['signal']
        label = int(data_dict['label'][0]-1)
        # data = signal.resample(data, 1000, axis=-1)
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
        train_files = os.listdir(os.path.join(self.datasets_dir, "processed_train"))
        val_files = os.listdir(os.path.join(self.datasets_dir, "processed_eval"))
        test_files = os.listdir(os.path.join(self.datasets_dir, "processed_test"))

        train_set = CustomDataset(os.path.join(self.datasets_dir, "processed_train"), train_files)
        val_set = CustomDataset(os.path.join(self.datasets_dir, "processed_eval"), val_files)
        test_set = CustomDataset(os.path.join(self.datasets_dir, "processed_test"), test_files)

        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))

        data_loader = {
            'train': make_train_loader(train_set, self.params, train_set.collate),
            'val': make_eval_loader(val_set, self.params, val_set.collate),
            'test': make_eval_loader(test_set, self.params, test_set.collate),
        }
        return data_loader
