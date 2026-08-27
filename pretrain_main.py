import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from datasets.pretraining_dataset import PretrainingDataset
from models.eeg_vision_pretrain import EEGAugment, EEGVisionPretrainModel
from pretrain_trainer import Trainer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_sampler(dataset, source_weights, balance_sources, samples_per_epoch):
    if source_weights and balance_sources:
        raise ValueError('Use either --source_weight or --balance_sources, not both')
    if source_weights and len(source_weights) != len(dataset.sources):
        raise ValueError('Repeat --source_weight exactly once per --dataset_dir')
    if balance_sources:
        source_weights = [1.0] * len(dataset.sources)
    if source_weights:
        if any(weight <= 0 for weight in source_weights):
            raise ValueError('--source_weight values must be positive')
        total = float(sum(source_weights))
        weights = []
        for source_weight, source_size in zip(source_weights, dataset.source_sizes):
            weights.extend([source_weight / total / source_size] * source_size)
        return WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            num_samples=samples_per_epoch or len(dataset),
            replacement=True,
        )
    if samples_per_epoch is not None:
        return RandomSampler(dataset, replacement=True, num_samples=samples_per_epoch)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='EEG-Vision VICReg pretraining')
    parser.add_argument('--dataset_dir', action='append', required=True,
                        help='LMDB or pickle directory; repeat for TUSZ and TUAB')
    parser.add_argument('--source_weight', action='append', type=float)
    parser.add_argument('--balance_sources', action='store_true')
    parser.add_argument('--samples_per_epoch', type=int, default=None)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--time_points', type=int, default=2000)

    parser.add_argument('--backbone_name', default='efficientnet_b0')
    parser.add_argument('--vision_fold_factor', type=int, default=8)
    parser.add_argument('--timm_pretrained', action=argparse.BooleanOptionalAction, default=True,
                        help='initialize the vision backbone from ImageNet weights')
    parser.add_argument('--online_weights', action='store_true')
    parser.add_argument('--projector_hidden_dim', type=int, default=512)
    parser.add_argument('--projector_dim', type=int, default=256)

    parser.add_argument('--amplitude_jitter', type=float, default=0.2)
    parser.add_argument('--noise_std', type=float, default=0.03)
    parser.add_argument('--channel_drop_prob', type=float, default=0.1)
    parser.add_argument('--time_mask_ratio', type=float, default=0.1)
    parser.add_argument('--max_time_shift', type=int, default=100)
    parser.add_argument('--sim_weight', type=float, default=25.0)
    parser.add_argument('--var_weight', type=float, default=25.0)
    parser.add_argument('--cov_weight', type=float, default=1.0)

    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--amp_dtype', choices=['float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--model_dir', default='experiments/checkpoints/vision_pretrain/tusz_tuab')
    parser.add_argument('--init_checkpoint', default=None,
                        help='initialize the vision backbone from a previous EEG checkpoint')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()


def main():
    params = parse_args()
    if params.batch_size < 2:
        raise ValueError('VICReg requires --batch_size >= 2')
    if params.vision_fold_factor <= 0 or params.time_points % params.vision_fold_factor:
        raise ValueError('--time_points must be divisible by --vision_fold_factor')
    if params.init_checkpoint and params.resume:
        raise ValueError('--init_checkpoint and --resume are mutually exclusive')
    if params.online_weights:
        os.environ['HF_HUB_OFFLINE'] = '0'
    else:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')

    print(params)
    setup_seed(params.seed)
    Path(params.model_dir).mkdir(parents=True, exist_ok=True)
    dataset = PretrainingDataset(
        params.dataset_dir,
        expected_shape=(params.channels, params.time_points),
    )
    print('Pretraining sources:')
    for source in dataset.describe():
        print(f"  {source['path']}: {source['samples']:,} {source['format']} samples")
    print(f'Total samples: {len(dataset):,}')

    sampler = make_sampler(dataset, params.source_weight, params.balance_sources, params.samples_per_epoch)
    use_cuda = params.device == 'cuda' or (params.device == 'auto' and torch.cuda.is_available())
    loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=sampler is None,
        sampler=sampler,
        pin_memory=use_cuda,
        persistent_workers=params.num_workers > 0,
        drop_last=not params.dry_run,
    )
    if len(loader) == 0:
        raise RuntimeError('No batch available; reduce --batch_size or add more samples')

    model = EEGVisionPretrainModel(
        backbone_name=params.backbone_name,
        fold_factor=params.vision_fold_factor,
        timm_pretrained=params.timm_pretrained,
        projector_hidden_dim=params.projector_hidden_dim,
        projector_dim=params.projector_dim,
    )
    augment = EEGAugment(
        amplitude_jitter=params.amplitude_jitter,
        noise_std=params.noise_std,
        channel_drop_prob=params.channel_drop_prob,
        time_mask_ratio=params.time_mask_ratio,
        max_time_shift=params.max_time_shift,
    )
    try:
        trainer = Trainer(params, loader, model, augment)
        trainer.dry_run() if params.dry_run else trainer.train()
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
