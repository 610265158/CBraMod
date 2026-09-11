"""Small timm-backbone compatibility layer for EEG vision models."""

import timm
import torch.nn.functional as F


def create_backbone(backbone_name, pretrained=True, drop_path_rate=0.0):
    return timm.create_model(
        backbone_name,
        pretrained=pretrained,
        in_chans=1,
        drop_path_rate=drop_path_rate,
    )


def encode_vision(eeg, adapter, backbone, feature_aggregation='gap'):
    """Encode folded EEG and aggregate the final spatial feature map."""
    image, chunk_shape = adapter(eeg)
    features = backbone.forward_features(image)
    if feature_aggregation == 'gap':
        features = _pool_features(features, backbone)
    elif feature_aggregation == 'cls_token':
        features = _cls_token_features(features)
    elif feature_aggregation == 'flatten':
        features = features.flatten(1)
    else:
        raise ValueError(
            'Unsupported vision feature aggregation: {}'.format(feature_aggregation)
        )
    return adapter.restore(features, chunk_shape)


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


def _cls_token_features(features):
    """Select the ViT classification token (index 0) from [B, N, C] tokens.

    timm DINOv3 ViTs (e.g. ``vit_small_patch16_dinov3``, instantiated as
    ``Eva``) default to ``global_pool='avg'``, so the ``gap`` branch averages
    only the patch tokens.  This branch exposes the final-normalized CLS token
    explicitly.
    """
    if features.ndim != 3:
        raise ValueError(
            'cls_token aggregation requires a 3D token tensor [B, N, C]; got {}'.format(
                tuple(features.shape)
            )
        )
    return features[:, 0]
