"""Lossless native-layout conversion around communication, preserving Node tensors."""
import math
import torch
from torch import nn


class NativeLayoutExchange(nn.Module):
    def __init__(self, exchange, axes):
        super().__init__(); self.exchange = exchange
        self.keys = exchange.keys; self.axes = [axes.get(k, 1) for k in self.keys]
        self.shapes = []
        for shape, axis in zip(exchange.shapes, self.axes):
            native = list(shape)
            if axis != 1:
                channel = native.pop(0); native.insert(axis % (len(shape)+1)-1, channel)
            self.shapes.append(tuple(native))
        self.lengths = [math.prod(s) for s in self.shapes]
        self.metadata = dict(exchange.metadata, native_channel_axes=dict(zip(self.keys, self.axes)))
        self.latest_inputs = self.latest_deltas = None

    @property
    def delta_only(self): return getattr(self.exchange, 'delta_only', False)

    @delta_only.setter
    def delta_only(self, value): self.exchange.delta_only = value

    def export_fixed_bases(self, directory): return self.exchange.export_fixed_bases(directory)

    def forward(self, *features):
        canonical = [x.movedim(axis, 1) if axis != 1 else x for x, axis in zip(features, self.axes)]
        packet = self.exchange(*canonical)
        self.latest_inputs = tuple(x.detach() for x in canonical)
        self.latest_deltas = getattr(self.exchange, 'latest_deltas', None)
        chunks = packet.split(self.exchange.lengths, 1)
        restored = []
        for x, shape, axis in zip(chunks, self.exchange.shapes, self.axes):
            x = x.reshape(-1, *shape)
            restored.append((x.movedim(1, axis) if axis != 1 else x).flatten(1))
        return torch.cat(restored, 1)
