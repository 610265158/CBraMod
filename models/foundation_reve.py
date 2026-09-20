"""REVE foundation model for the matched-pipeline rerun.

Wraps the released REVE-Base encoder with the released classifier pooling
options and an input adapter that consumes the repository's standard
``[B, C, T]`` batches:

1. the dataset loader is configured by ``finetune_main`` to emit samples
   normalised with the per-dataset scale of the released REVE task configs
   (1000 for FACED, 100 for TUAB/Mumtaz2016, 10 for HMC; no clipping); a 4D
   ``[B, S, C, T]`` batch (ISRUC) is flattened to ``[B * S, C, T]`` and the
   logits are folded back to ``[B, S, classes]``;
2. supply 3D electrode positions from the released position bank (bipolar
   channel names are averaged, following the released ``position_utils``);
3. classify with the released head geometry: ``pooling=no`` concatenates the
   attention-pooled context token with all patch tokens and applies an
   RMSNorm + linear head; ``pooling=last`` uses the pooled token only.

Weights are loaded directly from the released safetensors files (Hugging Face
cache snapshots by default), so no network access or remote code is required.
"""
from __future__ import annotations

import glob
import json
import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from configs.downstream import get_dataset_config
from configs.foundation import foundation_channels_and_time
from configs.foundation import foundation_spec

from .reve_backbone import RMSNorm
from .reve_encoder import REVE

REVE_EMBED_DIM = 512

DEFAULT_REVE_CONFIG = {
    'embed_dim': 512,
    'depth': 22,
    'heads': 8,
    'head_dim': 64,
    'mlp_dim_ratio': 2.66,
    'use_geglu': True,
    'freqs': 4,
    'patch_size': 200,
    'patch_overlap': 20,
    'noise_ratio': 0.0025,
}


class Model(nn.Module):
    """Stored [B, C, T] batch -> REVE encoder + released classifier geometry."""

    def __init__(self, param):
        super().__init__()
        dataset = get_dataset_config(param.downstream_dataset)
        spec = foundation_spec(param.downstream_dataset)
        reve = spec['reve']

        weights_dir = getattr(param, 'reve_weights_dir', None)
        cls_token = None
        state = None
        if getattr(param, 'use_pretrained_weights', True):
            config, state, cls_token = load_reve_checkpoint(weights_dir)
        else:
            config = dict(DEFAULT_REVE_CONFIG)
            print('REVE foundation checkpoint disabled; encoder is randomly initialized')

        backbone_args = SimpleNamespace(
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            heads=config['heads'],
            head_dim=config['head_dim'],
            mlp_dim_ratio=config['mlp_dim_ratio'],
            use_geglu=config['use_geglu'],
        )
        self.backbone = REVE(
            args_backbone=backbone_args,
            freqs=config['freqs'],
            patch_size=config['patch_size'],
            overlap_size=config['patch_overlap'],
            noise_ratio=config['noise_ratio'],
        )
        if state is not None:
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    'REVE checkpoint mismatch: missing {}, unexpected {}'.format(missing, unexpected)
                )
            print('REVE foundation checkpoint loaded: {}'.format(weights_dir or 'Hugging Face cache snapshot'))

        if cls_token is None:
            cls_token = torch.randn(1, 1, REVE_EMBED_DIM)
        self.cls_query_token = nn.Parameter(cls_token.reshape(1, 1, config['embed_dim']).clone())

        positions = load_positions(reve['electrodes'], getattr(param, 'reve_positions_dir', None))
        self.register_buffer('positions', positions, persistent=False)

        channels, time_steps = foundation_channels_and_time(dataset['input_shape'], param.downstream_dataset)
        self.num_of_patches = num_patches(time_steps, config['patch_size'], config['patch_overlap'])
        self.pooling = reve.get('pooling', 'last')
        self.num_of_classes = int(param.num_of_classes)
        dropout = float(getattr(param, 'dropout', reve.get('dropout', 0.1)))
        print(
            'REVE input adapter: {} channels x {} patches of {}, pooling={}, dropout={}; '
            'the loader applies microvolt / {} without clipping (released REVE convention)'.format(
                channels, self.num_of_patches, config['patch_size'], self.pooling, dropout,
                reve['input_scale'],
            )
        )
        self.head = _build_head(self.pooling, channels, self.num_of_patches, self.num_of_classes, dropout)

        self.frozen_backbone = bool(getattr(param, 'frozen', False))
        if self.frozen_backbone:
            self._freeze_backbone()

    def forward(self, eeg):
        chunks = None
        if eeg.ndim == 4:
            batch, chunks = eeg.shape[0], eeg.shape[1]
            eeg = eeg.reshape(batch * chunks, eeg.shape[2], eeg.shape[3])
        else:
            batch = eeg.shape[0]
        pos = self.positions.unsqueeze(0).repeat(eeg.shape[0], 1, 1).to(eeg.device)
        features = self.backbone(eeg, pos)

        query = self.cls_query_token.expand(eeg.shape[0], -1, -1)
        scores = torch.matmul(query, features.transpose(-1, -2)) / (features.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, features)
        if self.pooling == 'no':
            features = torch.cat([context, features], dim=-2)
        else:
            features = context.squeeze(1)

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
        print('Frozen REVE backbone parameters ({:,})'.format(count))


def num_patches(time_steps, patch_size, overlap_size):
    """Released REVE patch count (see get_flattened_output_dim)."""
    step = patch_size - overlap_size
    count = math.ceil((time_steps - patch_size) / step)
    if (time_steps - patch_size) % step == 0:
        count += 1
    return count


def _build_head(pooling, channels, patch_count, num_of_classes, dropout):
    if pooling == 'no':
        flat = (channels * patch_count + 1) * REVE_EMBED_DIM
        layers = [
            Rearrange('b n d -> b (n d)'),
            RMSNorm(flat),
            nn.Dropout(dropout),
            nn.Linear(flat, num_of_classes),
        ]
    elif pooling == 'last':
        layers = [
            RMSNorm(REVE_EMBED_DIM),
            nn.Dropout(dropout),
            nn.Linear(REVE_EMBED_DIM, num_of_classes),
        ]
    else:
        raise ValueError('Unsupported REVE pooling for the matched pipeline: {}'.format(pooling))
    if num_of_classes == 1:
        layers.append(Rearrange('b 1 -> b'))
    return nn.Sequential(*layers)


def hf_cache_root():
    for variable in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):
        value = os.environ.get(variable)
        if value:
            return value
    hf_home = os.environ.get('HF_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'))
    return os.path.join(hf_home, 'hub')


def resolve_hf_snapshot(repo_id, local_dir=None):
    if local_dir:
        if not os.path.exists(os.path.join(local_dir, 'config.json')):
            raise FileNotFoundError(
                'Expected config.json inside the supplied directory for {}: {}'.format(repo_id, local_dir)
            )
        return local_dir
    slug = 'models--' + repo_id.replace('/', '--')
    pattern = os.path.join(hf_cache_root(), slug, 'snapshots', '*')
    snapshots = glob.glob(pattern)
    if not snapshots:
        raise FileNotFoundError(
            'No Hugging Face cache snapshot for {} under {}. Download it, or pass an explicit directory.'
            .format(repo_id, pattern)
        )
    return max(snapshots, key=os.path.getmtime)


def load_reve_checkpoint(weights_dir=None):
    """Load the released REVE encoder weights from a safetensors or .pth file."""
    weights_dir = resolve_hf_snapshot('brain-bzh/reve-base', weights_dir)
    with open(os.path.join(weights_dir, 'config.json'), 'r', encoding='utf-8') as handle:
        config = json.load(handle)
    safetensors_path = os.path.join(weights_dir, 'model.safetensors')
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state = load_file(safetensors_path, device='cpu')
    else:
        candidates = sorted(glob.glob(os.path.join(weights_dir, '*.pth')))
        if not candidates:
            raise FileNotFoundError('No model.safetensors or .pth file found in {}'.format(weights_dir))
        state = torch.load(candidates[0], map_location='cpu')
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        state = {key.replace('module.', '').replace('encoder.', ''): value for key, value in state.items()}

    cls_token = state.pop('cls_query_token', None)
    if cls_token is None:
        raise RuntimeError('cls_query_token missing from the REVE checkpoint at {}'.format(weights_dir))
    return config, state, cls_token


def load_positions(electrode_names, positions_dir=None):
    """Released position bank lookup; bipolar names are averaged."""
    positions_dir = resolve_hf_snapshot('brain-bzh/reve-positions', positions_dir)
    with open(os.path.join(positions_dir, 'config.json'), 'r', encoding='utf-8') as handle:
        config = json.load(handle)
    names = list(config['position_names'])
    from safetensors.torch import load_file
    embedding = load_file(os.path.join(positions_dir, 'model.safetensors'), device='cpu')['embedding']
    mapping = {name: index for index, name in enumerate(names)}

    missing = []
    coordinates = []
    for name in electrode_names:
        parts = name.split('-') if '-' in name else [name]
        indices = []
        for part in parts:
            if part not in mapping:
                missing.append(part)
            else:
                indices.append(mapping[part])
        if len(indices) == len(parts):
            coordinates.append(embedding[indices].mean(dim=0))
    if missing:
        raise ValueError('Electrodes missing from the REVE position bank: {}'.format(sorted(set(missing))))
    print('REVE positions resolved for {} channels from {}'.format(len(coordinates), positions_dir))
    return torch.stack(coordinates)
