import torch


def validate_time_roll_params(params):
    probability = float(getattr(params, 'time_roll_prob', 1.0))
    max_fraction = float(getattr(params, 'time_roll_max_fraction', 0.5))
    if not 0.0 <= probability <= 1.0:
        raise ValueError('--time_roll_prob must be between 0 and 1')
    if not 0.0 <= max_fraction <= 0.5:
        raise ValueError('--time_roll_max_fraction must be between 0 and 0.5')
    return probability, max_fraction


def maybe_apply_time_roll(x, params):
    """Circularly shift each training sample along its time axis."""
    if not getattr(params, 'time_roll_augmentation', False):
        return x
    if x.ndim not in (3, 4):
        return x

    probability, max_fraction = validate_time_roll_params(params)
    time_length = x.shape[-1]
    max_shift = min(time_length // 2, int(round(time_length * max_fraction)))
    if probability <= 0.0 or max_shift <= 0:
        return x

    # Regular datasets sample one shift per item [B]. ISRUC samples one shift
    # per sleep epoch [B, S], while keeping all EEG channels synchronized.
    sample_shape = x.shape[:-2]
    # For an even-length signal, -T/2 and +T/2 are the same circular offset;
    # exclude the positive endpoint so every possible offset is sampled once.
    shift_high = max_shift + 1
    if time_length % 2 == 0 and max_shift == time_length // 2:
        shift_high = max_shift
    shifts = torch.randint(
        low=-max_shift,
        high=shift_high,
        size=sample_shape,
        device=x.device,
    )
    if probability < 1.0:
        apply_mask = torch.rand(sample_shape, device=x.device) < probability
        shifts = shifts * apply_mask

    source_index = (
        torch.arange(time_length, device=x.device)
        .reshape((1,) * len(sample_shape) + (time_length,))
        - shifts.unsqueeze(-1)
    ) % time_length
    source_index = source_index.unsqueeze(-2).expand(*x.shape[:-2], x.shape[-2], time_length)
    return torch.gather(x, dim=-1, index=source_index)
