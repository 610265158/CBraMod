import bisect
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
from torch.utils.data import Dataset

from utils.util import to_tensor


def _extract_eeg(value):
    if isinstance(value, dict):
        for key in ('X', 'signal', 'eeg', 'data'):
            if key in value:
                value = value[key]
                break
        else:
            raise KeyError('Pickle/LMDB dictionary has no X, signal, eeg, or data field')
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        array = array.reshape(array.shape[0], -1)
    return array


class _LmdbSource:
    def __init__(self, path):
        self.path = str(path)
        self._env = None
        self._pid = None
        env = self._open()
        try:
            with env.begin(write=False) as txn:
                value = txn.get(b'__keys__')
                if value is None:
                    raise RuntimeError(f'LMDB is missing __keys__: {self.path}')
                self.keys = pickle.loads(value)
        finally:
            env.close()

    def _open(self):
        return lmdb.open(
            self.path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512,
        )

    def _get_env(self):
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            if self._env is not None:
                self._env.close()
            self._env = self._open()
            self._pid = pid
        return self._env

    def __len__(self):
        return len(self.keys)

    def get(self, index):
        key = self.keys[index]
        encoded = key if isinstance(key, bytes) else str(key).encode()
        with self._get_env().begin(write=False) as txn:
            value = txn.get(encoded)
        if value is None:
            raise KeyError(f'Missing key {key!r} in {self.path}')
        return _extract_eeg(pickle.loads(value))

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_env'] = None
        state['_pid'] = None
        return state


class _PickleSource:
    def __init__(self, path):
        self.path = str(path)
        self.files = sorted(Path(path).rglob('*.pkl'))
        if not self.files:
            raise RuntimeError(f'No .pkl samples found under {self.path}')

    def __len__(self):
        return len(self.files)

    def get(self, index):
        with self.files[index].open('rb') as handle:
            return _extract_eeg(pickle.load(handle))

    def close(self):
        return None


def _is_lmdb(path):
    return (path / 'data.mdb').is_file() or path.suffix.lower() == '.lmdb'


class PretrainingDataset(Dataset):
    """Combine raw-wave LMDBs and/or pickle directories for vision SSL."""

    def __init__(self, dataset_dir, expected_shape=(16, 2000), validate_shape=True):
        super().__init__()
        paths = [dataset_dir] if isinstance(dataset_dir, (str, Path)) else list(dataset_dir)
        if not paths:
            raise ValueError('At least one --dataset_dir is required')
        self.expected_shape = tuple(expected_shape)
        self.sources = []
        for value in paths:
            path = Path(value).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f'Pretraining source does not exist: {path}')
            self.sources.append(_LmdbSource(path) if _is_lmdb(path) else _PickleSource(path))

        self.source_sizes = [len(source) for source in self.sources]
        self.cumulative_sizes = []
        total = 0
        for size in self.source_sizes:
            total += size
            self.cumulative_sizes.append(total)
        if total == 0:
            raise RuntimeError('Pretraining sources contain no samples')

        if validate_shape:
            for source in self.sources:
                shape = tuple(source.get(0).shape)
                if shape != self.expected_shape:
                    self.close()
                    raise ValueError(
                        f'Expected vision pretraining shape {self.expected_shape}, '
                        f'got {shape} from {source.path}'
                    )

    def __len__(self):
        return self.cumulative_sizes[-1]

    def source_index(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        source_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        previous = 0 if source_idx == 0 else self.cumulative_sizes[source_idx - 1]
        return source_idx, idx - previous

    def __getitem__(self, idx):
        source_idx, local_idx = self.source_index(idx)
        eeg = self.sources[source_idx].get(local_idx)
        if tuple(eeg.shape) != self.expected_shape:
            raise ValueError(
                f'Expected shape {self.expected_shape}, got {tuple(eeg.shape)} '
                f'from {self.sources[source_idx].path}'
            )
        return to_tensor(eeg)

    def describe(self):
        return [
            {'path': source.path, 'samples': size, 'format': source.__class__.__name__[1:-6].lower()}
            for source, size in zip(self.sources, self.source_sizes)
        ]

    def close(self):
        for source in getattr(self, 'sources', []):
            source.close()

    def __del__(self):
        self.close()
