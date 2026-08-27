import numpy as np


DEFAULT_EEG_CLIP_LIMIT = 1024
DEFAULT_EEG_SCALE_DIVISOR = 32.0


def clip_eeg(data, limit=DEFAULT_EEG_CLIP_LIMIT, scale=DEFAULT_EEG_SCALE_DIVISOR):
    data = np.asarray(data, dtype=np.float32)
    data = np.clip(data, -limit, limit)
    if scale is not None and scale != 1:
        data = data / np.float32(scale)
    return data


def robust_scale_eeg(data, clip=8.0, eps=1e-6):
    """Normalize one EEG trial with a global median/MAD robust scale.

    A single scale is shared by all channels, preserving their relative
    amplitudes while removing trial-level unit/scale mismatches.
    """
    data = np.asarray(data, dtype=np.float32)
    if clip <= 0:
        raise ValueError('robust EEG clip must be positive')
    center = np.median(data)
    mad = np.median(np.abs(data - center))
    robust_std = max(np.float32(1.4826) * mad, np.float32(eps))
    data = (data - center) / robust_std
    return np.clip(data, -clip, clip).astype(np.float32, copy=False)


def as_channel_time(x_data):
    if x_data.ndim == 4:
        batch_size, ch_num, patch_num, patch_size = x_data.shape
        return x_data.reshape(batch_size, ch_num, patch_num * patch_size)
    return x_data
