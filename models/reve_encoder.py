"""Vendored from REVE (https://github.com/elouayas/reve_eeg, MIT License).

REVE encoder with 4D Fourier positional encoding and patch embedding. Kept
semantically identical to the released code; the Hugging Face loader was
removed (weights are loaded directly from the released safetensors files in
``models.foundation_reve``).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from einops import rearrange
from torch import nn

from .reve_backbone import get_backbone


class REVE(nn.Module):
    def __init__(self, args_backbone, freqs, patch_size, overlap_size, noise_ratio):
        super().__init__()

        self.transformer = get_backbone(args_backbone)

        self.embed_dim = args_backbone.embed_dim

        self.freqs = freqs
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.noise_ratio = noise_ratio

        self.to_patch_embedding = patch_embedding(self.embed_dim, patch_size)
        self.fourier4d = FourierEmb4D(self.embed_dim, freqs=self.freqs)
        self.mlp4d = mlp_pos_embedding(self.embed_dim)
        self.ln = nn.LayerNorm(self.embed_dim)

    def forward(self, eeg, pos=None, return_output=False):
        device = eeg.device
        eeg = eeg.float()
        patches = eeg.unfold(dimension=2, size=self.patch_size, step=self.patch_size - self.overlap_size)
        b, c, h, p = patches.shape
        if self.training and pos is not None:  # add noise to the positions in training mode
            noise = torch.from_numpy(
                np.random.normal(loc=0, scale=self.noise_ratio, size=(c, 3))
            ).to(device=device, dtype=pos.dtype)
            pos = pos + noise
        pos = FourierEmb4D.add_time_patch(pos, h)
        pos_embed = self.ln(self.fourier4d(pos) + self.mlp4d(pos))
        x = rearrange(self.to_patch_embedding(patches), "b c h e -> b (c h) e", c=c, h=h,
                      e=self.embed_dim) + pos_embed
        x = self.transformer(x, return_output)
        return x

    def forward_attn(self, eeg, pos=None):
        device = eeg.device
        eeg = eeg.float()
        patches = eeg.unfold(dimension=2, size=self.patch_size, step=self.patch_size - self.overlap_size)
        b, c, h, p = patches.shape
        if self.training and pos is not None:  # add noise to the positions in training mode
            noise = torch.from_numpy(
                np.random.normal(loc=0, scale=self.noise_ratio, size=(c, 3))
            ).to(device=device, dtype=pos.dtype)
            pos = pos + noise
        pos = FourierEmb4D.add_time_patch(pos, h)
        pos_embed = self.ln(self.fourier4d(pos) + self.mlp4d(pos))
        x = rearrange(self.to_patch_embedding(patches), "b c h e -> b (c h) e", c=c, h=h,
                      e=self.embed_dim) + pos_embed
        x, attn = self.transformer.forward_attn(x)
        return x, attn


class Learnable4DPE(nn.Module):
    def __init__(self, embed_dim: int, positions: torch.Tensor, n_timesteps: int):
        super().__init__()
        self.embed_dim = embed_dim

        assert positions.dim() == 2 and positions.size(1) == 3, (
            "Positions should be a 2D tensor of shape (n_positions, 3)"
        )

        self.register_buffer("positions", positions)
        self.n_positions = len(positions)
        self.n_timesteps = n_timesteps

        self.spatial_pe = nn.Embedding(self.n_positions, self.embed_dim)
        self.temporal_pe = nn.Embedding(self.n_timesteps, self.embed_dim)

    def _convert_positions(self, pos: torch.Tensor, eps: float = 1e-5):
        """Turn a tensor of positions into a tensor of indices."""
        with torch.autocast("cuda" if pos.is_cuda else "cpu"):
            indices = torch.cdist(pos, self.positions.to(pos)).argmin(dim=-1)
            assert self.positions[indices].to(pos).allclose(pos, atol=eps), "Positions do not match"  # type: ignore
        return indices

    def forward(self, pos, n_timesteps: int | None = None):
        if n_timesteps is None:
            n_timesteps = self.n_timesteps
        time_indices = torch.arange(n_timesteps)

        B, C, _ = pos.shape
        pos = self._convert_positions(pos)
        spatial_pe = self.spatial_pe(pos)  # B, C, E
        temporal_pe = self.temporal_pe(time_indices)  # T, E

        spatial_pe = spatial_pe.unsqueeze(2).expand(-1, -1, n_timesteps, -1)  # B, C, T, E
        temporal_pe = temporal_pe.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)

        pe = spatial_pe + temporal_pe  # B, C, T, E

        pe = rearrange(pe, "b c t e -> b (c t) e")
        return pe


class FourierEmb4D(nn.Module):
    """
    Fourier positional embedding for 4D positions (x, y, z, t).
    This version allows for a reduced number of frequencies (n_freqs),
    and ensures the output embedding has the specified dimension.
    """

    def __init__(self, dimension: int, freqs: int, increment_time=0.1, margin: float = 0.4):
        super().__init__()
        self.dimension = dimension
        self.freqs = freqs
        self.increment_time = increment_time
        self.margin = margin

    def forward(self, positions_):
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        *U, _ = positions.shape

        freqs_w = torch.arange(self.freqs).to(positions)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        p_w = 2 * math.pi * freqs_w / width
        positions = positions[..., None, None, None, None, :]
        loc = (
            positions[..., 0] * p_x + positions[..., 1] * p_y + positions[..., 2] * p_z + positions[..., 3] * p_w
        ).view(*U, -1)
        if self.dimension != 512:  # noqa
            _, _, hd = loc.shape
            diff = hd - self.dimension // 2
            loc = loc[:, :, :-diff]
        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

    @classmethod
    def add_time_patch(cls, pos, num_patches):
        """
        Expand the position tensor by adding a time dimension, handling batched data.

        Args:
        - pos (Tensor): Input tensor of shape (B, C, 3), where B is the batch size,
        C is the number of channels, and 3 represents x, y, z.
        - num_patches (int): The number of time patches.

        Returns:
        - Tensor: Output tensor of shape (B, C * num_patches, 4), where each position is repeated with each time value.
        """
        B, C, _ = pos.shape
        pos_repeated = pos.unsqueeze(2).repeat(1, 1, num_patches, 1)  # Shape: (B, C, num_patches, 3)
        time_values = torch.arange(0, num_patches, 1, device=pos.device).float()  # Shape: (num_patches,)
        time_values = time_values.view(1, 1, num_patches, 1).expand(B, C, num_patches, 1)  # (B, C, num_patches, 1)
        pos_with_time = torch.cat((pos_repeated, time_values), dim=-1)  # Shape: (B, C, num_patches, 4)
        pos_with_time = pos_with_time.view(B, C * num_patches, 4)

        return pos_with_time


def patch_embedding(embed_dim, patch_size):
    to_patch_embedding = nn.Sequential(nn.Linear(patch_size, embed_dim))
    return to_patch_embedding


def mlp_pos_embedding(embed_dim):
    mlp_pos_embedding = nn.Sequential(nn.Linear(4, embed_dim, bias=False), nn.GELU(), nn.LayerNorm(embed_dim))
    return mlp_pos_embedding
