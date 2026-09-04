"""Small-feature nD reference Radon operator with an exact discrete transpose.

The dense matrix is intentional for the bounded pilot, not a full-resolution
production backend. Each row integrates multilinear native features over a
Householder-oriented hyperplane, with zero extension beyond the lattice.
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np
import torch
from torch import nn


def orientations(mesh):
    d = len(mesh) + 1
    if d == 1:
        return np.ones((1, 1)), np.ones(1)
    if any(m < 1 for m in mesh):
        raise ValueError("Every angular mesh size must be positive")
    angles = [np.arange(m) * math.pi / m for m in mesh]
    # Integrate the sphere's angular measure over nearest-sample parameter
    # cells. Positive cell weights also handle M=1 and repeated pole samples.
    roots, gw = np.polynomial.legendre.leggauss(16)
    axis_weights = []
    for j, a in enumerate(angles):
        bounds = np.r_[0., (a[:-1] + a[1:]) / 2, math.pi]
        w = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            u = (hi + lo) / 2 + roots * (hi - lo) / 2
            w.append(np.sum(gw * np.sin(u) ** (d - 2 - j)) * (hi - lo) / 2)
        axis_weights.append(w)
    directions, weights = [], []
    for idx in itertools.product(*[range(m) for m in mesh]):
        n, prod = [], 1.
        for j, m in enumerate(idx):
            n.append(prod * math.cos(angles[j][m]))
            prod *= math.sin(angles[j][m])
        n.append(prod)
        directions.append(n)
        weights.append(math.prod(axis_weights[j][m] for j, m in enumerate(idx)))
    weights = np.asarray(weights)
    return np.asarray(directions), weights / weights.sum()


def householder(n):
    e = np.zeros_like(n); e[0] = 1
    v = e - n
    if np.linalg.norm(v) < 1e-12:
        return np.eye(len(n))
    v /= np.linalg.norm(v)
    return np.eye(len(n)) - 2 * np.outer(v, v)


@lru_cache(maxsize=24)
def operator(shape, mesh, span):
    shape, mesh = tuple(shape), tuple(mesh)
    d = len(shape)
    if len(mesh) != d - 1 or span < 2 or min(shape) < 2:
        raise ValueError("Incompatible dimension, mesh, span or native shape")
    ns, weights = orientations(mesh)
    voxel_count = math.prod(shape)
    if len(ns) * span * voxel_count > 8_000_000:
        raise ValueError("Dense reference backend budget exceeded; use smaller pilot features")
    h = 2 / max(shape)
    # Includes the full support of the zero-extended multilinear basis.
    radius = np.linalg.norm((np.asarray(shape) + 1) * h / 2)
    s = np.linspace(-radius, radius, span)
    transverse_count = max(3, math.ceil(2 * radius / h) + 1)
    transverse = np.linspace(-radius, radius, transverse_count)
    axes = [s] + [transverse] * (d - 1)
    canonical = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, d)
    per_s = transverse_count ** (d - 1)
    row = np.repeat(np.arange(span), per_s)
    quadrature = (2 * radius / (transverse_count - 1)) ** (d - 1)
    # Endpoints of transverse integration lie beyond the native tent support,
    # except harmless zero-valued support-boundary samples.
    result = np.zeros((len(ns) * span, voxel_count), dtype=np.float64)
    strides = np.asarray([math.prod(shape[j + 1:]) for j in range(d)])
    for k, n in enumerate(ns):
        xyz = canonical @ householder(n).T
        index = xyz / h + (np.asarray(shape) - 1) / 2
        base = np.floor(index).astype(np.int64)
        fraction = index - base
        for corner in itertools.product((0, 1), repeat=d):
            corner = np.asarray(corner)
            ix = base + corner
            valid = ((ix >= 0) & (ix < shape)).all(axis=1)
            w = np.where(corner, fraction, 1 - fraction).prod(axis=1)
            np.add.at(result, (k * span + row[valid], ix[valid] @ strides),
                      w[valid] * quadrature * math.sqrt(weights[k]))
    # Fixed geometry-only operator-norm upper bound, not a learnable gate.
    scale = math.sqrt(result.sum(0).max() * result.sum(1).max())
    if scale <= 0:
        raise ValueError("Degenerate projector")
    return torch.from_numpy(result / scale), float(scale)


class Projector(nn.Module):
    def __init__(self, shape, mesh, span=16, scramble=False):
        super().__init__()
        a, self.scale = operator(tuple(shape), tuple(mesh), span)
        a = a.clone().float()
        if scramble:
            # Same singular values, sparsity, parameter count and compute;
            # only native spatial organization is destroyed.
            g = torch.Generator().manual_seed(971 + len(shape))
            a = a[:, torch.randperm(a.shape[1], generator=g)]
        self.register_buffer("matrix", a)
        self.shape, self.span = tuple(shape), span
        self.directions = math.prod(mesh)

    def forward(self, x):
        if tuple(x.shape[2:]) != self.shape:
            raise ValueError(f"Expected {self.shape}, got {tuple(x.shape[2:])}")
        z = x.flatten(2) @ self.matrix.T
        return z.reshape(x.shape[0], x.shape[1] * self.directions, self.span)

    def adjoint(self, z):
        z = z.reshape(z.shape[0], -1, self.directions * self.span)
        return (z @ self.matrix).reshape(z.shape[0], z.shape[1], *self.shape)


class Return(nn.Module):
    def __init__(self, projector):
        super().__init__(); self.projector = projector

    def forward(self, z):
        return self.projector.adjoint(z)
