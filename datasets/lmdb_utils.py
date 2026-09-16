import os

import lmdb


_ENV_CACHE = {}


def open_lmdb(data_dir):
    """Return a process-local LMDB environment for data_dir.

    The cache is keyed by process id so that forked DataLoader workers open
    their own environment instead of using the handle inherited from the main
    process (sharing one LMDB handle across fork is a known crash source in
    this pipeline).
    """
    data_dir = os.path.abspath(data_dir)
    key = (os.getpid(), data_dir)
    env = _ENV_CACHE.get(key)
    if env is None:
        env = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        _ENV_CACHE[key] = env
    return env
