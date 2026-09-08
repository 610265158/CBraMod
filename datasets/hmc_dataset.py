import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from datasets.sampling import make_eval_loader, make_train_loader
from datasets.shape_utils import clip_eeg
from utils.util import to_tensor


class CustomDataset(Dataset):
    def __init__(self, data_dir, mode="train"):
        mode_dir = os.path.join(data_dir, mode)
        if not os.path.isdir(mode_dir):
            raise FileNotFoundError(f"HMC split directory does not exist: {mode_dir}")
        self.files = sorted(
            os.path.join(mode_dir, name)
            for name in os.listdir(mode_dir)
            if name.endswith(".pkl")
        )
        if not self.files:
            raise RuntimeError(f"No HMC pickle files found in {mode_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            with open(path, "rb") as handle:
                sample = pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(f"Failed to load HMC sample: {path}") from exc
        data = np.asarray(sample["X"], dtype=np.float32)
        if data.shape != (4, 6000):
            raise ValueError(f"HMC sample {path} has shape {data.shape}, expected (4, 6000)")
        data = clip_eeg(data)
        label = int(np.asarray(sample["y"]).reshape(-1)[0])
        return data, label

    @staticmethod
    def collate(batch):
        x_data = np.stack([item[0] for item in batch], axis=0)
        y_data = np.asarray([item[1] for item in batch], dtype=np.int64)
        return to_tensor(x_data), to_tensor(y_data).long()


class LoadDataset:
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, "train")
        val_set = CustomDataset(self.datasets_dir, "val")
        test_set = CustomDataset(self.datasets_dir, "test")
        print(f"HMC samples: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        return {
            "train": make_train_loader(train_set, self.params, train_set.collate),
            "val": make_eval_loader(val_set, self.params, val_set.collate),
            "test": make_eval_loader(test_set, self.params, test_set.collate),
        }
