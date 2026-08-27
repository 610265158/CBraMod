from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.vision_backbone import load_backbone_checkpoint
from models.eeg_vision_pretrain import vicreg_loss


class Trainer:
    def __init__(self, params, data_loader, model, augment):
        self.params = params
        self.data_loader = data_loader
        self.device = self._resolve_device()
        self.model = model.to(self.device)
        self.augment = augment.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, params.epochs * len(data_loader)),
            eta_min=1e-6,
        )
        self.amp_enabled = params.amp and self.device.type == 'cuda'
        self.amp_dtype = torch.float16 if params.amp_dtype == 'float16' else torch.bfloat16
        self.scaler = torch.amp.GradScaler(
            'cuda', enabled=self.amp_enabled and self.amp_dtype == torch.float16,
        )
        self.start_epoch = 0
        self.best_loss = float('inf')
        if params.init_checkpoint:
            self._load_backbone(params.init_checkpoint)
        if params.resume:
            self._resume(params.resume)
        print(
            f'Device: {self.device}; backbone: {params.backbone_name}; '
            f'AMP: {self.amp_enabled} ({params.amp_dtype}); batches/epoch: {len(data_loader):,}'
        )

    def _resolve_device(self):
        if self.params.device == 'cpu':
            return torch.device('cpu')
        if not torch.cuda.is_available():
            if self.params.device == 'cuda':
                raise RuntimeError('--device cuda requested, but CUDA is unavailable')
            return torch.device('cpu')
        if not 0 <= self.params.cuda < torch.cuda.device_count():
            raise ValueError(f'Invalid CUDA index {self.params.cuda}')
        return torch.device(f'cuda:{self.params.cuda}')

    def _load_backbone(self, path):
        load_backbone_checkpoint(self.model.backbone, path)
        print(f'Initialized EEG vision backbone from {path}')

    def _resume(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = int(checkpoint['epoch']) + 1
        self.best_loss = float(checkpoint.get('best_loss', float('inf')))
        print(f'Resumed {path} at epoch {self.start_epoch + 1}')

    def _prepare(self, eeg):
        eeg = eeg.to(self.device, non_blocking=True)
        eeg = torch.clamp(eeg, -1024.0, 1024.0) / 32.0
        if not torch.isfinite(eeg).all():
            raise ValueError('Input contains NaN or Inf')
        return eeg

    def _forward_loss(self, eeg):
        eeg = self._prepare(eeg)
        view1 = self.augment(eeg.clone())
        view2 = self.augment(eeg.clone())
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            z1 = self.model(view1)
            z2 = self.model(view2)
        loss, parts = vicreg_loss(
            z1,
            z2,
            sim_weight=self.params.sim_weight,
            var_weight=self.params.var_weight,
            cov_weight=self.params.cov_weight,
        )
        return eeg, z1, loss, parts

    def dry_run(self):
        self.model.train()
        eeg = next(iter(self.data_loader))
        with torch.no_grad():
            scaled, embedding, loss, parts = self._forward_loss(eeg)
        memory = (
            torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
            if self.device.type == 'cuda' else 0.0
        )
        print(
            f'Dry run OK: input={tuple(eeg.shape)}, scaled_range='
            f'[{scaled.min().item():.4f}, {scaled.max().item():.4f}], '
            f'embedding={tuple(embedding.shape)}, loss={loss.item():.6f}, '
            f'inv={parts["invariance"].item():.6f}, var={parts["variance"].item():.6f}, '
            f'cov={parts["covariance"].item():.6f}, peak_cuda_memory={memory:.1f} MiB'
        )

    def _save(self, epoch, mean_loss, is_best):
        output = Path(self.params.model_dir)
        checkpoint = {
            'epoch': epoch,
            'best_loss': min(self.best_loss, mean_loss),
            'model_state_dict': self.model.state_dict(),
            'backbone_state_dict': self.model.backbone.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'params': vars(self.params),
        }
        torch.save(checkpoint, output / 'last_checkpoint.pth')
        if is_best:
            torch.save(self.model.backbone.state_dict(), output / 'best_backbone.pth')
            print(f'Backbone saved in {output / "best_backbone.pth"}')

    def train(self):
        for epoch in range(self.start_epoch, self.params.epochs):
            self.model.train()
            losses = []
            part_sums = {'invariance': 0.0, 'variance': 0.0, 'covariance': 0.0}
            for eeg in tqdm(self.data_loader, mininterval=10):
                self.optimizer.zero_grad(set_to_none=True)
                _, _, loss, parts = self._forward_loss(eeg)
                self.scaler.scale(loss).backward()
                if self.params.clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                losses.append(loss.detach().float().item())
                for key in part_sums:
                    part_sums[key] += parts[key].float().item()

            mean_loss = float(np.mean(losses))
            divisor = max(1, len(losses))
            print(
                f'Epoch {epoch + 1}: loss={mean_loss:.6f}, '
                f'inv={part_sums["invariance"] / divisor:.6f}, '
                f'var={part_sums["variance"] / divisor:.6f}, '
                f'cov={part_sums["covariance"] / divisor:.6f}, '
                f'lr={self.optimizer.param_groups[0]["lr"]:.8f}'
            )
            is_best = mean_loss < self.best_loss
            self._save(epoch, mean_loss, is_best)
            if is_best:
                self.best_loss = mean_loss
