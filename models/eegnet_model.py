"""Compact EEGNet-8,2 baseline for the downstream benchmark suite."""

import torch
import torch.nn as nn

from configs.downstream import get_dataset_config


class Model(nn.Module):
    """EEGNet with temporal, depthwise-spatial, and separable convolutions.

    Regular inputs have shape [B, C, T]. ISRUC inputs have shape
    [B, S, C, T]; its sequence dimension is restored before the loss.
    """

    def __init__(self, param):
        super().__init__()
        config = get_dataset_config(param.downstream_dataset)
        input_shape = config['input_shape']
        self.is_sequence = len(input_shape) == 3
        channels, samples = input_shape[-2:]

        f1 = 8
        depth_multiplier = 2
        f2 = f1 * depth_multiplier
        temporal_kernel = 64
        separable_kernel = 16
        dropout = float(getattr(param, 'dropout', 0.5))

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, temporal_kernel),
                padding='same',
                bias=False,
            ),
            nn.BatchNorm2d(f1, eps=1e-3, momentum=0.01),
            nn.Conv2d(
                f1,
                f2,
                kernel_size=(channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f2, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                f2,
                f2,
                kernel_size=(1, separable_kernel),
                padding='same',
                groups=f2,
                bias=False,
            ),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            feature_dim = self.features(dummy).numel()
        self.classifier = nn.Linear(feature_dim, param.num_of_classes)
        self.squeeze_binary = param.num_of_classes == 1
        self._init_weights()

        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        print(
            'EEGNet-8,2: channels={}, samples={}, feature_dim={}, parameters={:,}'.format(
                channels,
                samples,
                feature_dim,
                parameter_count,
            )
        )

    def forward(self, eeg):
        sequence_shape = None
        if eeg.ndim == 4:
            batch_size, sequence_length, channels, samples = eeg.shape
            sequence_shape = (batch_size, sequence_length)
            eeg = eeg.reshape(batch_size * sequence_length, channels, samples)
        if eeg.ndim != 3:
            raise ValueError('EEGNet expects [B,C,T] or [B,S,C,T], got {}'.format(tuple(eeg.shape)))

        features = self.features(eeg.unsqueeze(1)).flatten(1)
        logits = self.classifier(features)
        if sequence_shape is not None:
            logits = logits.reshape(*sequence_shape, -1)
        if self.squeeze_binary and logits.size(-1) == 1:
            logits = logits[..., 0]
        return logits

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
