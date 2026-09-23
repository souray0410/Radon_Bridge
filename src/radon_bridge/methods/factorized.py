"""Globally factorized, bias-free convolution in projected feature space."""
import torch
from torch import nn


class FactorizedMixer(nn.Module):
    def __init__(self, widths, rank, kernel_size=3):
        super().__init__()
        self.widths = tuple(widths)
        total = sum(widths)
        if type(rank) is not int or not 1 <= rank <= total:
            raise ValueError('Global bottleneck rank must be an integer in [1, sum(CM)]')
        if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError('Positive odd kernel required')
        self.encoder = nn.Conv1d(total, rank, 1, bias=False)
        self.decoder = nn.Conv1d(rank, total, 1, bias=False)
        self.conv = nn.Conv1d(rank, rank, kernel_size, padding=kernel_size//2, bias=False)
        nn.init.zeros_(self.conv.weight)
        self.register_buffer('mask', torch.ones_like(self.conv.weight))

    def forward(self, *features):
        z = self.encoder(torch.cat(features, dim=1))
        return self.decoder(self.conv(z)).split(self.widths, dim=1)

    def dense_weight(self, maximum_elements=1_000_000):
        total = sum(self.widths)
        if total * total * self.conv.kernel_size[0] > maximum_elements:
            raise ValueError('Dense export exceeds verification budget')
        return torch.einsum('or,rst,si->oit', self.decoder.weight[:, :, 0],
                            self.conv.weight, self.encoder.weight[:, :, 0])
