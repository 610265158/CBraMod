"""EEG-to-image folding used by downstream and pretraining models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseFoldAdapter(nn.Module):
    """Convert 3D or 4D EEG into one phase-interleaved folded image.

    For fold factor ``P`` and ``W = T // P``, output row ``c * P + p``
    contains ``eeg[..., c, p::P]``.  This is the original
    ``view(C, W, P).permute(C, P, W)`` operation: it is a bijective sample
    permutation, not a split into ``P`` contiguous time chunks.
    """

    def __init__(
            self,
            fold_factor=4,
            pad_multiple=None,
    ):
        super().__init__()
        if fold_factor < 1:
            raise ValueError('Vision folding requires fold_factor >= 1; got {}.'.format(fold_factor))
        self.fold_factor = fold_factor
        self.pad_multiple = pad_multiple

    def forward(self, eeg):
        image, chunk_shape = self._flatten_chunks(eeg)
        image = self._fold_time(image)
        if self.pad_multiple:
            image = _pad_to_multiple(image, self.pad_multiple)
        return image, chunk_shape

    def extra_repr(self):
        return 'fold_factor={}, pad_multiple={}'.format(self.fold_factor, self.pad_multiple)

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
        # [B,1,C,T] -> [B,1,C,W,P] -> [B,1,C,P,W].  Consequently,
        # grid[b,0,c,p,w] == image[b,0,c,w * P + p].
        grid = image.reshape(batch, 1, channels, width, self.fold_factor)
        grid = grid.permute(0, 1, 2, 4, 3).contiguous()
        return grid.reshape(batch, 1, channels * self.fold_factor, width)

    @staticmethod
    def restore(features, chunk_shape):
        if chunk_shape is None:
            return features
        batch, chunks = chunk_shape
        return features.reshape(batch, chunks, -1)


def repeat_eeg_channels(eeg, repeats):
    if repeats == 1:
        return eeg
    if eeg.ndim not in (3, 4):
        raise ValueError(
            'Channel repetition expects [B,C,T] or [B,S,C,T], got {}'.format(tuple(eeg.shape))
        )
    return torch.repeat_interleave(eeg, repeats=repeats, dim=eeg.ndim - 2)


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
