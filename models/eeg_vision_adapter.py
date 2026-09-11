"""EEG-to-image folding used by the downstream vision model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseFoldAdapter(nn.Module):
    """Convert 3D or 4D EEG into one folded image.

    For fold factor ``P`` and ``W = T // P``, the ``phase`` mode (default)
    makes output row ``c * P + p`` contain ``eeg[..., c, p::P]``.  This is the
    original ``view(C, W, P).permute(C, P, W)`` operation: it is a bijective
    sample permutation, not a split into ``P`` contiguous time chunks.

    Every other mode keeps the identical output shape ``[B, 1, C * P, W]`` and
    serves as a geometry control for the phase-interleaved layout:
      - ``chunk``: row ``c * P + p`` contains the contiguous chunk
        ``eeg[..., c, p * W:(p + 1) * W]`` (a naive reshape).
      - ``phase_shuffle``: ``phase`` with a fixed random permutation of the
        ``P`` phase rows.
      - ``channel_shuffle``: ``phase`` with a fixed random permutation of the
        ``C`` channel rows.
    """

    MODES = ('phase', 'chunk', 'phase_shuffle', 'channel_shuffle')

    def __init__(self, fold_factor=4, pad_multiple=None, mode='phase', shuffle_seed=7):
        super().__init__()
        if fold_factor < 1:
            raise ValueError('Vision folding requires fold_factor >= 1; got {}.'.format(fold_factor))
        if mode not in self.MODES:
            raise ValueError('Unknown fold mode {!r}; expected one of {}'.format(mode, self.MODES))
        self.fold_factor = fold_factor
        self.pad_multiple = pad_multiple
        self.mode = mode
        self.shuffle_seed = shuffle_seed
        self._phase_perm = None
        self._channel_perm = None
        if mode == 'phase_shuffle':
            self._phase_perm = _shuffle_indices(fold_factor, shuffle_seed)

    def forward(self, eeg):
        image, chunk_shape = self._flatten_chunks(eeg)
        image = self._fold_time(image)
        if self.pad_multiple:
            image = _pad_to_multiple(image, self.pad_multiple)
        return image, chunk_shape

    def extra_repr(self):
        return 'fold_factor={}, pad_multiple={}, mode={}'.format(
            self.fold_factor, self.pad_multiple, self.mode)

    def _flatten_chunks(self, eeg):
        if eeg.ndim == 3:
            batch, channels, _ = eeg.shape
            return eeg.reshape(batch, 1, channels, -1), None
        if eeg.ndim == 4:
            batch, chunks, channels, _ = eeg.shape
            return eeg.reshape(batch * chunks, 1, channels, -1), (batch, chunks)
        raise ValueError(
            'Expected EEG shape [B,C,T] or [B,S,C,T], got {}'.format(tuple(eeg.shape))
        )

    def _fold_time(self, image):
        batch, _, channels, time_points = image.shape
        if time_points % self.fold_factor:
            raise ValueError(
                'Time length {} must be divisible by fold factor {}.'.format(
                    time_points, self.fold_factor
                )
            )

        width = time_points // self.fold_factor

        if self.mode == 'channel_shuffle':
            image = self._shuffle_channels(image)

        if self.mode == 'chunk':
            grid = image.reshape(batch, 1, channels, self.fold_factor, width)
            return grid.reshape(batch, 1, channels * self.fold_factor, width)

        # [B,1,C,T] -> [B,1,C,W,P] -> [B,1,C,P,W].  Consequently,
        # grid[b,0,c,p,w] == image[b,0,c,w * P + p].
        grid = image.reshape(batch, 1, channels, width, self.fold_factor)
        grid = grid.permute(0, 1, 2, 4, 3).contiguous()
        if self.mode == 'phase_shuffle':
            grid = grid[:, :, :, self._phase_perm.to(grid.device), :]
        return grid.reshape(batch, 1, channels * self.fold_factor, width)

    def _shuffle_channels(self, image):
        channels = image.shape[2]
        if self._channel_perm is None or len(self._channel_perm) != channels:
            self._channel_perm = _shuffle_indices(channels, self.shuffle_seed)
        return image[:, :, self._channel_perm.to(image.device), :]

    @staticmethod
    def restore(features, chunk_shape):
        if chunk_shape is None:
            return features
        batch, chunks = chunk_shape
        return features.reshape(batch, chunks, -1)


def _shuffle_indices(size, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(size, generator=generator)


def _pad_to_multiple(image, multiple):
    height_multiple, width_multiple = (multiple, multiple) if isinstance(multiple, int) else multiple
    if height_multiple < 1 or width_multiple < 1:
        raise ValueError('Padding multiples must be positive; got {}.'.format(multiple))

    height, width = image.shape[-2:]
    pad_height = (-height) % height_multiple
    pad_width = (-width) % width_multiple
    # Keep the folded EEG anchored at the top-left.  Padding is appended only
    # after the final phase row and time column so it never shifts the physical
    # origin seen by the convolutional backbone.
    padding = (
        0,
        pad_width,
        0,
        pad_height,
    )
    return F.pad(image, padding, mode='constant', value=0)
