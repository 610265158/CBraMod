import torch
import torch.nn as nn
import torch.nn.functional as F

from .eeg_vision_adapter import PhaseFoldAdapter
from .vision_backbone import create_backbone, encode_vision


class EEGVisionPretrainModel(nn.Module):
    """VICReg encoder built from the same adapter/backbone used downstream."""

    def __init__(
            self,
            backbone_name='efficientnet_b0',
            fold_factor=8,
            timm_pretrained=True,
            projector_hidden_dim=512,
            projector_dim=256,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.adapter = PhaseFoldAdapter(fold_factor=fold_factor)
        self.backbone = create_backbone(backbone_name, pretrained=timm_pretrained)
        feature_dim = self.backbone.num_features
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projector_hidden_dim, bias=False),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, projector_hidden_dim, bias=False),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, projector_dim),
        )

    def encode(self, x):
        return encode_vision(x, self.adapter, self.backbone)

    def forward(self, x):
        return self.projector(self.encode(x))


class EEGAugment(nn.Module):
    """Label-free augmentations that preserve coarse EEG physiology."""

    def __init__(
            self,
            amplitude_jitter=0.2,
            noise_std=0.03,
            channel_drop_prob=0.1,
            time_mask_ratio=0.1,
            max_time_shift=100,
    ):
        super().__init__()
        self.amplitude_jitter = amplitude_jitter
        self.noise_std = noise_std
        self.channel_drop_prob = channel_drop_prob
        self.time_mask_ratio = time_mask_ratio
        self.max_time_shift = max_time_shift

    def forward(self, x):
        batch_size, channels, time_points = x.shape
        if self.amplitude_jitter > 0:
            scale = torch.empty(batch_size, 1, 1, device=x.device).uniform_(
                1.0 - self.amplitude_jitter, 1.0 + self.amplitude_jitter,
            )
            x = x * scale
        if self.noise_std > 0:
            sample_std = x.float().std(dim=-1, keepdim=True).clamp_min(1e-4).to(x.dtype)
            x = x + torch.randn_like(x) * sample_std * self.noise_std
        if self.channel_drop_prob > 0:
            keep = torch.rand(batch_size, channels, 1, device=x.device) >= self.channel_drop_prob
            x = x * keep.to(x.dtype)
        if self.time_mask_ratio > 0:
            width = max(1, int(round(time_points * self.time_mask_ratio)))
            starts = torch.randint(0, max(1, time_points - width + 1), (batch_size,), device=x.device)
            time = torch.arange(time_points, device=x.device).view(1, 1, -1)
            masked = (time >= starts.view(-1, 1, 1)) & (time < (starts + width).view(-1, 1, 1))
            x = x.masked_fill(masked, 0)
        if self.max_time_shift > 0:
            shifts = torch.randint(
                -self.max_time_shift, self.max_time_shift + 1, (batch_size,), device=x.device,
            )
            base = torch.arange(time_points, device=x.device).view(1, -1)
            indices = (base - shifts.view(-1, 1)) % time_points
            x = torch.gather(x, 2, indices[:, None, :].expand(-1, channels, -1))
        return x


def vicreg_loss(z1, z2, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    z1 = z1.float()
    z2 = z2.float()
    invariance = F.mse_loss(z1, z2)

    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)
    std1 = torch.sqrt(z1_centered.var(dim=0, unbiased=False) + 1e-4)
    std2 = torch.sqrt(z2_centered.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (F.relu(1 - std1).mean() + F.relu(1 - std2).mean())

    denominator = max(1, z1.shape[0] - 1)
    cov1 = z1_centered.T @ z1_centered / denominator
    cov2 = z2_centered.T @ z2_centered / denominator
    covariance = (_off_diagonal(cov1).pow(2).sum() + _off_diagonal(cov2).pow(2).sum()) / z1.shape[1]
    total = sim_weight * invariance + var_weight * variance + cov_weight * covariance
    return total, {
        'invariance': invariance.detach(),
        'variance': variance.detach(),
        'covariance': covariance.detach(),
    }


def _off_diagonal(matrix):
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()
