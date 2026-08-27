import math

import torch


def validate_amplitude_scale_params(params):
    probability = float(getattr(params, 'amplitude_scale_prob', 1.0))
    min_scale = float(getattr(params, 'amplitude_scale_min', 0.5))
    max_scale = float(getattr(params, 'amplitude_scale_max', 2.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError('--amplitude_scale_prob must be between 0 and 1')
    if min_scale <= 0.0:
        raise ValueError('--amplitude_scale_min must be positive')
    if max_scale < min_scale:
        raise ValueError('--amplitude_scale_max must be >= --amplitude_scale_min')
    return probability, min_scale, max_scale


def maybe_apply_amplitude_scale(x, params):
    """Multiply each training sample by an independent log-uniform scale."""
    if not getattr(params, 'amplitude_scale_augmentation', False):
        return x
    if x.ndim not in (3, 4):
        return x

    probability, min_scale, max_scale = validate_amplitude_scale_params(params)
    if probability <= 0.0 or min_scale == max_scale == 1.0:
        return x

    sample_shape = x.shape[:-2]
    if min_scale == max_scale:
        scales = x.new_full(sample_shape, min_scale)
    else:
        log_min = math.log(min_scale)
        log_max = math.log(max_scale)
        scales = torch.empty(sample_shape, device=x.device, dtype=x.dtype).uniform_(log_min, log_max).exp_()

    if probability < 1.0:
        apply_mask = torch.rand(sample_shape, device=x.device) < probability
        scales = torch.where(apply_mask, scales, torch.ones_like(scales))

    return x * scales.reshape(*sample_shape, 1, 1)
