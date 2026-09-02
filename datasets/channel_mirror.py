import re

import torch


SELF_MIRROR_CHANNELS = {'A2-A1'}


BIPOLAR_16_CHANNELS = (
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
)

STANDARD_32_CHANNELS = (
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8',
    'FC1', 'FC2', 'FC5', 'FC6',
    'CZ', 'C3', 'C4', 'T3', 'T4', 'A1', 'A2',
    'CP1', 'CP2', 'CP5', 'CP6',
    'PZ', 'P3', 'P4', 'T5', 'T6',
    'PO3', 'PO4', 'OZ', 'O1', 'O2',
)

SEEDV_62_CHANNELS = (
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2',
)

PHYSIONET_64_CHANNELS = (
    'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'FP1', 'FPZ', 'FP2',
    'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
    'O1', 'OZ', 'O2', 'IZ',
)

BCIC2020_3_64_CHANNELS = (
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
    'P7', 'P3', 'PZ', 'P4', 'P8',
    'PO9', 'O1', 'OZ', 'O2', 'PO10',
    'AF7', 'AF3', 'AF4', 'AF8',
    'F5', 'F1', 'F2', 'F6',
    'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10',
    'C5', 'C1', 'C2', 'C6',
    'TP7', 'CP3', 'CPZ', 'CP4', 'TP8',
    'P5', 'P1', 'P2', 'P6',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
)

MUMTAZ_19_CHANNELS = (
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ',
)

MENTAL_ARITHMETIC_20_CHANNELS = (
    'FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4',
    'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2',
    'FZ', 'CZ', 'PZ', 'A2-A1',
)

ISRUC_6_CHANNELS = (
    'F3-A2', 'C3-A2', 'O1-A2',
    'F4-A1', 'C4-A1', 'O2-A1',
)


DATASET_CHANNEL_ORDERS = {
    'CHB-MIT': BIPOLAR_16_CHANNELS,
    'TUAB': BIPOLAR_16_CHANNELS,
    'TUEV': BIPOLAR_16_CHANNELS,
    'ISRUC': ISRUC_6_CHANNELS,
    'FACED': STANDARD_32_CHANNELS,
    'SEED-V': SEEDV_62_CHANNELS,
    'PhysioNet-MI': PHYSIONET_64_CHANNELS,
    'SHU-MI': STANDARD_32_CHANNELS,
    'BCIC2020-3': BCIC2020_3_64_CHANNELS,
    'Mumtaz2016': MUMTAZ_19_CHANNELS,
    'MentalArithmetic': MENTAL_ARITHMETIC_20_CHANNELS,
}


def normalize_channel_name(name):
    return name.upper().replace(' ', '')


def mirror_electrode_name(name):
    normalized = normalize_channel_name(name)
    match = re.match(r'^(.*?)(\d+)([^0-9]*)$', normalized)
    if match is None:
        return normalized
    prefix, number_text, suffix = match.groups()
    number = int(number_text)
    mirrored = number + 1 if number % 2 else number - 1
    return '{}{}{}'.format(prefix, mirrored, suffix)


def mirror_channel_name(name):
    normalized = normalize_channel_name(name)
    if normalized in SELF_MIRROR_CHANNELS:
        return normalized
    return '-'.join(mirror_electrode_name(part) for part in normalized.split('-'))


def build_mirror_permutation(channel_names):
    by_name = {normalize_channel_name(name): index for index, name in enumerate(channel_names)}
    permutation = []
    for name in channel_names:
        source_name = mirror_channel_name(name)
        if source_name not in by_name:
            raise ValueError(
                'Mirrored channel {} for {} is not present in the channel order.'.format(
                    source_name,
                    name,
                )
            )
        permutation.append(by_name[source_name])
    return tuple(permutation)


DATASET_MIRROR_PERMUTATIONS = {
    name: build_mirror_permutation(channel_names)
    for name, channel_names in DATASET_CHANNEL_ORDERS.items()
}


def channel_mirror_permutation(dataset_name, channel_count):
    permutation = DATASET_MIRROR_PERMUTATIONS.get(dataset_name)
    if permutation is None:
        return None
    if len(permutation) != channel_count:
        return None
    return permutation


def maybe_apply_channel_mirror(x, params):
    if not getattr(params, 'mirror_augmentation', False):
        return x

    probability = float(getattr(params, 'mirror_prob', 0.5))
    if probability <= 0:
        return x
    if x.ndim not in (3, 4):
        return x

    channel_dim = 1 if x.ndim == 3 else 2
    permutation = channel_mirror_permutation(
        getattr(params, 'downstream_dataset', None),
        x.shape[channel_dim],
    )
    if permutation is None:
        return x

    batch_size = x.shape[0]
    apply_mask = torch.rand(batch_size, device=x.device) < probability
    if not bool(apply_mask.any()):
        return x

    permutation = torch.as_tensor(permutation, dtype=torch.long, device=x.device)
    x = x.clone()
    if x.ndim == 3:
        x[apply_mask] = x[apply_mask].index_select(1, permutation)
    else:
        x[apply_mask] = x[apply_mask].index_select(2, permutation)
    return x


def maybe_apply_channel_mirror_with_label_swap(x, y, params):
    """Mirror SHU-MI channels and swap binary left/right labels together.

    SHU-MI motor-imagery labels are side-specific: a left/right montage mirror
    changes the semantic class, so the corresponding binary label must flip.
    This helper intentionally lives beside the generic channel mirror path and
    is only called for datasets whose labels are known to have this symmetry.
    """
    if not getattr(params, 'mirror_augmentation', False):
        return x, y

    probability = float(getattr(params, 'mirror_prob', 0.5))
    if probability <= 0 or x.ndim not in (3, 4):
        return x, y

    channel_dim = 1 if x.ndim == 3 else 2
    permutation = channel_mirror_permutation(
        getattr(params, 'downstream_dataset', None),
        x.shape[channel_dim],
    )
    if permutation is None:
        return x, y

    batch_size = x.shape[0]
    apply_mask = torch.rand(batch_size, device=x.device) < probability
    if not bool(apply_mask.any()):
        return x, y

    permutation = torch.as_tensor(permutation, dtype=torch.long, device=x.device)
    x = x.clone()
    if x.ndim == 3:
        x[apply_mask] = x[apply_mask].index_select(1, permutation)
    else:
        x[apply_mask] = x[apply_mask].index_select(2, permutation)

    y = y.clone()
    y[apply_mask] = 1 - y[apply_mask]
    return x, y
