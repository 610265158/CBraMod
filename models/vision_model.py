"""Phase-folded EEG classifier with a timm vision backbone."""

import torch
import torch.nn as nn

from configs.downstream import get_dataset_config
from configs.backbones import backbone_name_for, load_backbone_config
from datasets.shape_utils import DEFAULT_EEG_SCALE_DIVISOR

from .eeg_vision_adapter import PhaseFoldAdapter, repeat_eeg_channels
from .vision_backbone import configure_height_stride, create_backbone, encode_vision
from .vision_backbone import load_backbone_checkpoint


class Model(nn.Module):
    """Phase fold -> vision backbone -> configured feature aggregation -> head."""

    def __init__(self, param):
        super().__init__()
        dataset_config = get_dataset_config(param.downstream_dataset)
        config = dataset_config['vision']
        profile = load_backbone_config(
            getattr(param, 'backbone_config', None),
            dataset=param.downstream_dataset,
        )
        profile_vision = profile.get('vision', {})
        if profile_vision:
            config.update(profile_vision)
            if 'adapter' in profile_vision:
                config['adapter'] = dict(dataset_config['vision'].get('adapter', {}))
                config['adapter'].update(profile_vision['adapter'])
        for key, attr in (
            ('feature_aggregation', 'vision_feature_aggregation'),
            ('init_head', 'vision_init_head'),
            ('squeeze_binary', 'vision_squeeze_binary'),
        ):
            value = getattr(param, attr, None)
            if value is not None:
                config[key] = value
        fold_factor = getattr(param, 'vision_fold_factor', None)
        if fold_factor is None:
            fold_factor = config['adapter']['fold_factor']

        pad_multiple = None
        if not bool(getattr(param, 'vision_no_pad', False)):
            pad_multiple = (getattr(param, 'vision_height_stride', 32), 32)
        self.adapter = PhaseFoldAdapter(
            fold_factor=fold_factor,
            pad_multiple=pad_multiple,
        )
        self._configure_input(param)

        backbone_name = (
            getattr(param, 'backbone_name', None)
            or backbone_name_for(profile, config['backbone_name'])
        )
        self.backbone = create_backbone(
            backbone_name,
            pretrained=getattr(param, 'use_pretrained_weights', True),
            drop_path_rate=getattr(param, 'drop_path_rate', 0.0),
        )
        checkpoint = getattr(param, 'vision_pretrained_checkpoint', None)
        if checkpoint:
            load_backbone_checkpoint(self.backbone, checkpoint)
            print('Loaded EEG-Vision pretrained backbone: {}'.format(checkpoint))

        configure_height_stride(
            self.backbone,
            getattr(param, 'vision_height_stride', 32),
            backbone_name,
        )
        self.frozen_backbone = bool(getattr(param, 'frozen', False))
        if self.frozen_backbone:
            self._freeze_backbone()

        self.feature_aggregation = config.get('feature_aggregation', 'gap')
        if self.feature_aggregation == 'gap':
            feature_dim = _feature_dim(self.backbone)
        elif self.feature_aggregation == 'flatten':
            feature_dim = _infer_flat_feature_dim(
                self.backbone,
                self.adapter,
                dataset_config['input_shape'],
                self.channel_repeat,
            )
        else:
            raise ValueError(
                'Unsupported vision feature aggregation: {}'.format(
                    self.feature_aggregation
                )
            )
        self.dropout = nn.Dropout(getattr(param, 'dropout', 0.1))
        self.head = nn.Linear(feature_dim, param.num_of_classes)
        if config.get('init_head', True):
            head_init_std = getattr(param, 'vision_head_init_std', None)
            if head_init_std is None:
                head_init_std = config.get('head_init_std')
            if head_init_std is not None:
                if head_init_std <= 0:
                    raise ValueError('--vision_head_init_std must be positive')
                nn.init.trunc_normal_(self.head.weight, std=head_init_std)
                nn.init.zeros_(self.head.bias)
            else:
                head_init = getattr(param, 'vision_head_init', None) or config.get('head_init', 'trunc_normal')
                _init_head(self.head, head_init)
        self.squeeze_binary = config.get('squeeze_binary', False)

    def forward(self, eeg):
        eeg = self._prepare_input(eeg)
        features = self.dropout(encode_vision(
            eeg,
            self.adapter,
            self.backbone,
            feature_aggregation=self.feature_aggregation,
        ))
        logits = self.head(features)
        if self.squeeze_binary and logits.size(-1) == 1:
            return logits[..., 0]
        return logits

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_backbone:
            self.backbone.eval()
        return self

    def _configure_input(self, param):
        self.dataset_mean = getattr(param, 'eeg_dataset_mean', None)
        self.dataset_std = getattr(param, 'eeg_dataset_std', None)
        self.target_std = getattr(param, 'eeg_target_std', 1.0)
        self.channel_repeat = getattr(param, 'vision_channel_repeat', 1)

        if (self.dataset_mean is None) != (self.dataset_std is None):
            raise ValueError('--eeg_dataset_mean and --eeg_dataset_std must be set together')
        if self.dataset_std is not None and self.dataset_std <= 0:
            raise ValueError('--eeg_dataset_std must be positive')
        if self.target_std <= 0:
            raise ValueError('--eeg_target_std must be positive')
        if self.channel_repeat < 1:
            raise ValueError('--vision_channel_repeat must be at least 1')

    def _prepare_input(self, eeg):
        if self.dataset_std is not None:
            raw_eeg = eeg * DEFAULT_EEG_SCALE_DIVISOR
            eeg = self.target_std * (raw_eeg - self.dataset_mean) / self.dataset_std
        return repeat_eeg_channels(eeg, self.channel_repeat)

    def _freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        parameter_count = sum(parameter.numel() for parameter in self.backbone.parameters())
        print('Linear probe: froze backbone parameters ({:,})'.format(parameter_count))


def _feature_dim(backbone):
    classifier = backbone.get_classifier() if hasattr(backbone, 'get_classifier') else None
    return getattr(classifier, 'in_features', None) or backbone.num_features


def _infer_flat_feature_dim(backbone, adapter, input_shape, channel_repeat):
    """Infer a fixed flatten-head width without materializing a LazyLinear."""
    dummy = torch.zeros((1,) + tuple(input_shape), dtype=torch.float32)
    dummy = repeat_eeg_channels(dummy, channel_repeat)
    image, _ = adapter(dummy)

    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        features = backbone.forward_features(image)
    backbone.train(was_training)

    feature_dim = features.flatten(1).shape[1]
    print(
        'Flatten vision head: final feature map {} -> {} FC inputs'.format(
            tuple(features.shape[1:]), feature_dim
        )
    )
    return feature_dim


def _init_head(head, mode):
    if mode == 'trunc_normal':
        nn.init.trunc_normal_(head.weight, std=0.02)
        nn.init.zeros_(head.bias)
    elif mode == 'small_trunc_normal':
        nn.init.trunc_normal_(head.weight, std=1e-3)
        nn.init.zeros_(head.bias)
    elif mode == 'rare_binary_prior':
        if head.out_features != 1:
            raise ValueError('rare_binary_prior requires a single-logit binary head')
        nn.init.trunc_normal_(head.weight, std=1e-3)
        nn.init.constant_(head.bias, -5.01)
    elif mode == 'zero':
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    elif mode == 'xavier_uniform':
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
    else:
        raise ValueError('Unsupported vision head initialization: {}'.format(mode))
