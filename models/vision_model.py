"""EEG classifier built from a phase-folding adapter and a timm backbone."""

import torch
import torch.nn as nn

from configs.backbones import backbone_name_for, load_backbone_config
from configs.downstream import get_dataset_config

from .eeg_vision_adapter import PhaseFoldAdapter
from .vision_backbone import create_backbone, encode_vision


class Model(nn.Module):
    """Input -> phase folding -> timm encoder -> dropout -> linear head."""

    def __init__(self, param):
        super().__init__()
        dataset = get_dataset_config(param.downstream_dataset)
        config, profile = _vision_config(dataset, param)

        fold_factor = getattr(param, 'vision_fold_factor', None)
        fold_factor = config['adapter']['fold_factor'] if fold_factor is None else fold_factor
        self.adapter = PhaseFoldAdapter(
            fold_factor=fold_factor,
            pad_multiple=None if getattr(param, 'vision_no_pad', False) else (32, 32),
        )

        backbone_name = getattr(param, 'backbone_name', None) or backbone_name_for(
            profile, config['backbone_name']
        )
        self.backbone = create_backbone(
            backbone_name,
            pretrained=getattr(param, 'use_pretrained_weights', True),
            drop_path_rate=getattr(param, 'drop_path_rate', 0.0),
        )
        self.frozen_backbone = bool(getattr(param, 'frozen', False))
        if self.frozen_backbone:
            self._freeze_backbone()

        self.feature_aggregation = config.get('feature_aggregation', 'gap')
        if self.feature_aggregation == 'gap':
            classifier = self.backbone.get_classifier() if hasattr(self.backbone, 'get_classifier') else None
            feature_dim = getattr(classifier, 'in_features', None) or self.backbone.num_features
        elif self.feature_aggregation == 'flatten':
            feature_dim = _flat_feature_dim(self.backbone, self.adapter, dataset['input_shape'])
        else:
            raise ValueError('Unsupported vision feature aggregation: {}'.format(self.feature_aggregation))

        self.dropout = nn.Dropout(getattr(param, 'dropout', 0.1))
        self.head = nn.Linear(feature_dim, param.num_of_classes)
        _initialize_head(self.head, config, param)
        self.squeeze_binary = bool(config.get('squeeze_binary', False))

    def forward(self, eeg):
        features = encode_vision(
            eeg, self.adapter, self.backbone,
            feature_aggregation=self.feature_aggregation,
        )
        logits = self.head(self.dropout(features))
        return logits[..., 0] if self.squeeze_binary and logits.size(-1) == 1 else logits

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_backbone:
            self.backbone.eval()
        return self

    def _freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        count = sum(parameter.numel() for parameter in self.backbone.parameters())
        print('Linear probe: froze backbone parameters ({:,})'.format(count))


def _vision_config(dataset, param):
    config = dict(dataset['vision'])
    profile = load_backbone_config(
        getattr(param, 'backbone_config', None), dataset=param.downstream_dataset
    )
    profile_vision = profile.get('vision', {})
    config.update(profile_vision)
    if 'adapter' in profile_vision:
        config['adapter'] = {
            **dataset['vision'].get('adapter', {}), **profile_vision['adapter']
        }
    for key, attr in (
        ('feature_aggregation', 'vision_feature_aggregation'),
        ('squeeze_binary', 'vision_squeeze_binary'),
    ):
        value = getattr(param, attr, None)
        if value is not None:
            config[key] = value
    return config, profile


def _flat_feature_dim(backbone, adapter, input_shape):
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros((1,) + tuple(input_shape))
        features = backbone.forward_features(adapter(dummy)[0])
    backbone.train(was_training)
    feature_dim = features.flatten(1).shape[1]
    print('Flatten vision head: final feature map {} -> {} FC inputs'.format(
        tuple(features.shape[1:]), feature_dim
    ))
    return feature_dim


def _initialize_head(head, config, param):
    # Some locked recipes intentionally keep PyTorch's default Linear init.
    if not config.get('init_head', True):
        return
    std = getattr(param, 'vision_head_init_std', None)
    if std is None:
        std = config.get('head_init_std', 0.02)
    if std <= 0:
        raise ValueError('--vision_head_init_std must be positive')
    nn.init.trunc_normal_(head.weight, std=std)
    nn.init.zeros_(head.bias)
