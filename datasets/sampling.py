import os
import pickle
import struct
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.amplitude_scale import (
    amplitude_scale_distribution,
    maybe_apply_amplitude_scale,
    validate_amplitude_scale_params,
)
from datasets.channel_mirror import (
    DATASET_MIRROR_PERMUTATIONS,
    maybe_apply_channel_mirror,
    maybe_apply_channel_mirror_with_label_swap,
)
from datasets.time_roll import maybe_apply_time_roll, validate_time_roll_params


def make_train_loader(dataset, params, collate_fn):
    sampler = make_balanced_sampler(dataset, params)
    collate_fn = make_train_collate_fn(collate_fn, params)
    kwargs = {
        'batch_size': params.batch_size,
        'collate_fn': collate_fn,
        **loader_runtime_kwargs(params),
    }
    if sampler is None:
        kwargs['shuffle'] = True
    else:
        kwargs['sampler'] = sampler
        kwargs['shuffle'] = False
    return DataLoader(dataset, **kwargs)


def make_train_collate_fn(collate_fn, params):
    mirror_enabled = getattr(params, 'mirror_augmentation', False)
    mirror_probability = float(getattr(params, 'mirror_prob', 0.5))
    if mirror_enabled and mirror_probability > 0:
        dataset_name = getattr(params, 'downstream_dataset', None)
        if dataset_name not in DATASET_MIRROR_PERMUTATIONS:
            print('Channel mirror augmentation skipped: no permutation configured for {}'.format(dataset_name))
            mirror_enabled = False
        else:
            print(
                'Channel mirror augmentation enabled: dataset={}, prob={}'.format(
                    dataset_name,
                    mirror_probability,
                )
            )

    time_roll_enabled = getattr(params, 'time_roll_augmentation', False)
    if time_roll_enabled:
        probability, max_fraction = validate_time_roll_params(params)
        time_roll_enabled = probability > 0 and max_fraction > 0
        if time_roll_enabled:
            print(
                'Time roll augmentation enabled: prob={}, max_fraction={}'.format(
                    probability,
                    max_fraction,
                )
            )

    amplitude_scale_enabled = getattr(params, 'amplitude_scale_augmentation', False)
    if amplitude_scale_enabled:
        probability, min_scale, max_scale = validate_amplitude_scale_params(params)
        distribution = amplitude_scale_distribution(params)
        amplitude_scale_enabled = probability > 0 and not (min_scale == max_scale == 1.0)
        if amplitude_scale_enabled:
            print(
                'Amplitude scale augmentation enabled: prob={}, range=[{}, {}], distribution={}'.format(
                    probability,
                    min_scale,
                    max_scale,
                    distribution,
                )
            )

    if not mirror_enabled and not time_roll_enabled and not amplitude_scale_enabled:
        return collate_fn

    def wrapped_collate(batch):
        x, y = collate_fn(batch)
        if mirror_enabled:
            if dataset_name == 'SHU-MI':
                x, y = maybe_apply_channel_mirror_with_label_swap(x, y, params)
            else:
                x = maybe_apply_channel_mirror(x, params)
        if time_roll_enabled:
            x = maybe_apply_time_roll(x, params)
        if amplitude_scale_enabled:
            x = maybe_apply_amplitude_scale(x, params)
        return x, y

    return wrapped_collate


def make_eval_loader(dataset, params, collate_fn, batch_size=None):
    return DataLoader(
        dataset,
        batch_size=batch_size or params.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        **loader_runtime_kwargs(params),
    )


def loader_runtime_kwargs(params):
    """Fast loader settings that do not change sampling or model numerics."""
    num_workers = getattr(params, 'num_workers', 0)
    device = str(getattr(params, 'device', 'cpu'))
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': device.startswith('cuda'),
    }
    if num_workers > 0:
        # Avoid paying worker startup and dataset/LMDB initialization on every
        # epoch. Two prefetched batches per worker is PyTorch's conservative
        # default and keeps host-memory growth bounded.
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    return kwargs


def make_balanced_sampler(dataset, params):
    if not getattr(params, 'balanced_sampling', False):
        return None
    if getattr(params, 'downstream_task', None) == 'regression':
        print('Balanced train sampling skipped for regression task.')
        return None

    sample_labels = load_sample_labels(dataset)
    if not sample_labels:
        print('Balanced train sampling skipped: labels are unavailable.')
        return None
    if len(sample_labels) != len(dataset):
        raise ValueError(
            'Balanced sampler label count mismatch: labels={}, dataset={}'.format(
                len(sample_labels),
                len(dataset),
            )
        )

    counts = Counter()
    for labels in sample_labels:
        counts.update(labels)
    if len(counts) <= 1:
        print('Balanced train sampling skipped: only one class found.')
        return None

    negative_ratio = float(getattr(params, 'balanced_sampling_negative_ratio', 0.0))
    if negative_ratio < 0:
        raise ValueError('--balanced_sampling_negative_ratio must be non-negative')
    if negative_ratio > 0:
        if set(counts) != {0, 1} or any(len(labels) != 1 for labels in sample_labels):
            raise ValueError('negative-ratio sampling requires binary single-label samples')
        positive_indices = [i for i, labels in enumerate(sample_labels) if labels[0] == 1]
        negative_indices = [i for i, labels in enumerate(sample_labels) if labels[0] == 0]
        requested_negatives = int(round(negative_ratio * len(positive_indices)))
        if not positive_indices or not negative_indices:
            print('Negative-ratio sampling skipped: one class is empty.')
            return None
        sample_count = len(positive_indices) + min(len(negative_indices), requested_negatives)
        generator = torch.Generator()
        generator.manual_seed(getattr(params, 'seed', 3407))
        sampler = RatioClassSampler(
            positive_indices,
            negative_indices,
            negative_count=min(len(negative_indices), requested_negatives),
            num_samples=sample_count,
            generator=generator,
        )
        print(
            'Negative-ratio sampling enabled: positives={}, negatives={}, target_ratio=1:{}, samples_per_epoch={}'.format(
                len(positive_indices),
                len(negative_indices),
                negative_ratio,
                sample_count,
            )
        )
        return sampler

    min_share = float(getattr(params, 'balanced_sampling_min_share', 0.0))
    if min_share < 0 or min_share >= 1:
        raise ValueError('--balanced_sampling_min_share must be in [0, 1)')

    if min_share > 0:
        if any(len(labels) != 1 for labels in sample_labels):
            raise ValueError('minimum-share sampling currently requires one label per sample')
        target_shares = minimum_class_shares(counts, min_share)
        weights = [target_shares[labels[0]] / counts[labels[0]] for labels in sample_labels]
        sampling_description = 'min_share={}, targets={}'.format(
            min_share,
            ', '.join('{}={:.4f}'.format(label, target_shares[label]) for label in sorted(target_shares)),
        )
    else:
        power = float(getattr(params, 'balanced_sampling_power', 1.0))
        if power < 0 or power > 1:
            raise ValueError('--balanced_sampling_power must be in [0, 1]')
        weights = []
        for labels in sample_labels:
            if labels:
                weights.append(sum(counts[label] ** (-power) for label in labels) / len(labels))
            else:
                weights.append(0.0)
        sampling_description = 'power={}'.format(power)

    generator = torch.Generator()
    generator.manual_seed(getattr(params, 'seed', 3407))
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    print('Balanced train sampling enabled: {}, {}'.format(
        sampling_description,
        format_counts(counts),
    ))
    return sampler


class RatioClassSampler(torch.utils.data.Sampler):
    """Sample every positive and a fixed number of negatives each epoch."""

    def __init__(self, positive_indices, negative_indices, negative_count, num_samples, generator):
        self.positive_indices = tuple(positive_indices)
        self.negative_indices = tuple(negative_indices)
        self.negative_count = int(negative_count)
        self.num_samples = int(num_samples)
        self.generator = generator

    def __iter__(self):
        pos_order = torch.randperm(len(self.positive_indices), generator=self.generator).tolist()
        neg_order = torch.randperm(len(self.negative_indices), generator=self.generator).tolist()
        selected = [self.positive_indices[i] for i in pos_order]
        selected.extend(self.negative_indices[i] for i in neg_order[:self.negative_count])
        order = torch.randperm(len(selected), generator=self.generator).tolist()
        return iter([selected[i] for i in order])

    def __len__(self):
        return self.num_samples


def minimum_class_shares(counts, min_share):
    """Floor rare classes at min_share and redistribute the remainder by prior."""
    labels = sorted(counts)
    total = float(sum(counts.values()))
    natural = {label: counts[label] / total for label in labels}
    floored = [label for label in labels if natural[label] < min_share]
    remaining = [label for label in labels if label not in floored]
    floor_mass = min_share * len(floored)
    if floor_mass >= 1 or not remaining:
        raise ValueError(
            'minimum class share {} is incompatible with {} classes'.format(min_share, len(labels))
        )
    remaining_prior = sum(natural[label] for label in remaining)
    targets = {label: min_share for label in floored}
    for label in remaining:
        targets[label] = (1.0 - floor_mass) * natural[label] / remaining_prior
    return targets


def load_sample_labels(dataset):
    if hasattr(dataset, 'get_labels'):
        return normalize_sample_labels(dataset.get_labels())

    if hasattr(dataset, 'db') and hasattr(dataset, 'keys'):
        return labels_from_lmdb(dataset)

    if hasattr(dataset, 'files'):
        return labels_from_pickle_files(dataset)

    if hasattr(dataset, 'seqs_labels_path_pair'):
        return labels_from_npy_pairs(dataset)

    return []


def labels_from_lmdb(dataset):
    labels = []
    with dataset.db.begin(write=False) as txn:
        for key in dataset.keys:
            pair = pickle.loads(txn.get(key.encode()))
            labels.append(normalize_one_sample(pair['label']))
    return labels


def labels_from_pickle_files(dataset):
    labels = []
    for filename in dataset.files:
        path = filename
        if not os.path.isabs(path) and hasattr(dataset, 'data_dir'):
            path = os.path.join(dataset.data_dir, filename)
        label = fast_pickle_tail_label(path)
        if label is None:
            with open(path, 'rb') as handle:
                data_dict = pickle.load(handle)
            if 'y' in data_dict:
                label = data_dict['y']
            elif 'label' in data_dict:
                label = data_dict['label']
            else:
                raise KeyError('Cannot find label key in {}'.format(path))
        labels.append(normalize_one_sample(label))
    return labels


def fast_pickle_tail_label(path, tail_size=512):
    try:
        with open(path, 'rb') as handle:
            size = os.fstat(handle.fileno()).st_size
            handle.seek(max(0, size - tail_size))
            tail = handle.read()
    except OSError:
        return None

    label = parse_pickle_int_after_key(tail, b'\x8c\x01y')
    if label is not None:
        return label

    label = parse_pickle_int_after_key(tail, b'\x8c\x05label')
    if label is not None:
        return label

    return parse_pickle_float64_ndarray_after_key(tail, b'\x8c\x05label')


def parse_pickle_int_after_key(tail, key):
    key_pos = tail.rfind(key)
    if key_pos < 0:
        return None
    pos = key_pos + len(key)
    while pos < len(tail) and tail[pos] in (0x94,):
        pos += 1
    if pos >= len(tail):
        return None

    opcode = tail[pos]
    if opcode == ord('K') and pos + 1 < len(tail):
        return tail[pos + 1]
    if opcode == ord('M') and pos + 2 < len(tail):
        return int.from_bytes(tail[pos + 1:pos + 3], byteorder='little', signed=False)
    if opcode == ord('J') and pos + 4 < len(tail):
        return int.from_bytes(tail[pos + 1:pos + 5], byteorder='little', signed=True)
    return None


def parse_pickle_float64_ndarray_after_key(tail, key):
    key_pos = tail.rfind(key)
    if key_pos < 0:
        return None
    data_pos = tail.find(b'C\x08', key_pos)
    if data_pos < 0 or data_pos + 10 > len(tail):
        return None
    return int(struct.unpack('<d', tail[data_pos + 2:data_pos + 10])[0])


def labels_from_npy_pairs(dataset):
    labels = []
    for _, label_path in dataset.seqs_labels_path_pair:
        labels.append(normalize_one_sample(np.load(label_path)))
    return labels


def normalize_sample_labels(sample_labels):
    return [normalize_one_sample(label) for label in sample_labels]


def normalize_one_sample(label):
    array = np.asarray(label)
    if array.ndim == 0:
        return [int(array.item())]
    return [int(value) for value in array.reshape(-1)]


def format_counts(counts):
    return ', '.join(
        '{}={}'.format(label, counts[label])
        for label in sorted(counts)
    )
