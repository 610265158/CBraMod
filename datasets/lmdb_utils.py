import os

import lmdb


_ENV_CACHE = {}


def open_lmdb(data_dir):
    data_dir = os.path.abspath(data_dir)
    env = _ENV_CACHE.get(data_dir)
    if env is None:
        env = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        _ENV_CACHE[data_dir] = env
    return env
