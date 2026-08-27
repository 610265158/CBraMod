"""Small timm-backbone compatibility layer for EEG vision models."""

from math import prod

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_backbone(backbone_name, pretrained=True, drop_path_rate=0.0):
    return timm.create_model(
        backbone_name,
        pretrained=pretrained,
        in_chans=1,
        drop_path_rate=drop_path_rate,
    )


def encode_vision(eeg, adapter, backbone):
    """Encode folded EEG with the backbone's global average pooling."""
    image, chunk_shape = adapter(eeg)
    features = backbone.forward_features(image)
    features = _pool_features(features, backbone)
    return adapter.restore(features, chunk_shape)


def load_backbone_checkpoint(backbone, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Vision checkpoint must contain a state dict: {}'.format(checkpoint_path))

    state = checkpoint.get('backbone_state_dict', checkpoint.get('model_state_dict', checkpoint))
    for prefix in ('module.', 'backbone.', 'model.'):
        if any(key.startswith(prefix) for key in state):
            state = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
    backbone.load_state_dict(state, strict=True)


def configure_height_stride(backbone, target_stride, backbone_name):
    """Reduce CNN height stride without changing its time stride."""
    strided_convs = [
        (name, layer)
        for name, layer in backbone.named_modules()
        if isinstance(layer, nn.Conv2d) and layer.stride[0] > 1
    ]
    current_stride = prod(layer.stride[0] for _, layer in strided_convs)
    if target_stride == current_stride:
        return
    if target_stride < 1 or current_stride % target_stride:
        raise ValueError(
            'Cannot set height stride {} for {} with convolutional height stride {}.'.format(
                target_stride, backbone_name, current_stride
            )
        )

    remaining_reduction = current_stride // target_stride
    changed = []
    for name, layer in reversed(strided_convs):
        height_stride, time_stride = layer.stride
        if remaining_reduction == 1:
            break
        if remaining_reduction % height_stride:
            continue
        layer.stride = (1, time_stride)
        remaining_reduction //= height_stride
        changed.append(name)

    if remaining_reduction != 1:
        raise ValueError(
            'Unable to reduce {} height stride from {} to {}.'.format(
                backbone_name, current_stride, target_stride
            )
        )
    print(
        'Configured {} height stride: {} -> {}; kept time stride unchanged; modified {}'.format(
            backbone_name, current_stride, target_stride, list(reversed(changed))
        )
    )


def _pool_features(features, backbone):
    if hasattr(backbone, 'forward_head'):
        features = backbone.forward_head(features, pre_logits=True)
        return features.flatten(1) if features.ndim > 2 else features
    if features.ndim == 4:
        return F.adaptive_avg_pool2d(features, 1).flatten(1)
    if features.ndim == 3:
        return features[:, 0] if getattr(backbone, 'num_prefix_tokens', 0) else features.mean(dim=1)
    if features.ndim == 2:
        return features
    raise ValueError('Unsupported feature shape: {}'.format(tuple(features.shape)))
