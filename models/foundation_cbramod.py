"""CBraMod foundation model for the matched-pipeline rerun.

Wraps the released CBraMod encoder with a task head and an input adapter that
consumes the repository's standard ``[B, C, T]`` batches.  The dataset loader
is configured by ``finetune_main`` to emit samples normalised with the released
CBraMod convention (microvolt / 100, no clipping); the adapter then

1. reshapes ``[B, C, T]`` to the ``[B, C, S, 200]`` patch layout; a 4D
   ``[B, S, C, T]`` batch (ISRUC) is flattened to ``[B * S, C, T]`` first and
   the logits are folded back to ``[B, S, classes]``;
2. classifies with the released per-dataset head variants
   (``all_patch_reps``; the two-layer variant is used when the
   ``S * 200`` hidden width would exceed 2,000, as for HMC's 30 patches).
"""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from configs.downstream import get_dataset_config
from configs.foundation import foundation_channels_and_time
from configs.foundation import foundation_spec

from .cbramod import CBraMod

PATCH_SIZE = 200
MAX_THREE_LAYER_HIDDEN = 2000


class Model(nn.Module):
    """Stored [B, C, T] batch -> CBraMod patch layout -> head."""

    def __init__(self, param):
        super().__init__()
        dataset = get_dataset_config(param.downstream_dataset)
        spec = foundation_spec(param.downstream_dataset)

        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8,
        )
        checkpoint = getattr(param, 'foundation_dir', None) or 'pretrained_weights/pretrained_weights.pth'
        if getattr(param, 'use_pretrained_weights', True):
            try:
                state = torch.load(checkpoint, map_location='cpu')
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'CBraMod foundation checkpoint not found at {}. Download it from '
                    'https://huggingface.co/weighting666/CBraMod (see pretrained_weights/README.md).'
                    .format(checkpoint)
                ) from exc
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    'CBraMod checkpoint mismatch at {}: missing {}, unexpected {}'.format(
                        checkpoint, missing, unexpected
                    )
                )
            print('CBraMod foundation checkpoint loaded: {}'.format(checkpoint))
        else:
            print('CBraMod foundation checkpoint disabled; backbone is randomly initialized')
        self.backbone.proj_out = nn.Identity()

        channels, time_steps = foundation_channels_and_time(dataset['input_shape'], param.downstream_dataset)
        if time_steps % PATCH_SIZE:
            raise ValueError(
                'CBraMod requires the time axis to be divisible by {}; dataset {} has {}'.format(
                    PATCH_SIZE, param.downstream_dataset, time_steps
                )
            )
        self.patch_count = time_steps // PATCH_SIZE
        print(
            'CBraMod input adapter: {} channels x {} patches of {}; the loader applies microvolt / {} '
            'without clipping (released CBraMod convention)'.format(
                channels, self.patch_count, PATCH_SIZE, spec['cbramod']['input_scale'],
            )
        )

        self.num_of_classes = int(param.num_of_classes)
        self.head = _build_head(channels, self.patch_count, self.num_of_classes,
                                dropout=float(getattr(param, 'dropout', 0.1)))

        self.frozen_backbone = bool(getattr(param, 'frozen', False))
        if self.frozen_backbone:
            self._freeze_backbone()

    def forward(self, eeg):
        chunks = None
        if eeg.ndim == 4:
            batch, chunks = eeg.shape[0], eeg.shape[1]
            eeg = eeg.reshape(batch * chunks, eeg.shape[2], eeg.shape[3])
        x = eeg.reshape(eeg.shape[0], eeg.shape[1], self.patch_count, PATCH_SIZE)
        features = self.backbone(x)
        logits = self.head(features)
        if chunks is not None:
            logits = logits.reshape(batch, chunks, -1)
        return logits[..., 0] if self.num_of_classes == 1 and logits.size(-1) == 1 else logits

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_backbone:
            self.backbone.eval()
        return self

    def _freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        count = sum(parameter.numel() for parameter in self.backbone.parameters())
        print('Frozen CBraMod backbone parameters ({:,})'.format(count))


def _build_head(channels, patch_count, num_of_classes, dropout):
    """Released CBraMod head variants (``all_patch_reps`` / two-layer)."""
    flat = channels * patch_count * PATCH_SIZE
    hidden = patch_count * PATCH_SIZE
    if hidden <= MAX_THREE_LAYER_HIDDEN:
        layers = [
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(flat, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, num_of_classes),
        ]
    else:
        print(
            'CBraMod head: S*P={} exceeds {}; using the released two-layer variant'.format(
                hidden, MAX_THREE_LAYER_HIDDEN
            )
        )
        layers = [
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(flat, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, num_of_classes),
        ]
    if num_of_classes == 1:
        layers.append(Rearrange('b 1 -> b'))
    return nn.Sequential(*layers)
