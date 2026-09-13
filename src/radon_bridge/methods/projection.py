"""Fixed n-D antipodal EEM, raw hyperplane Radon, and direct backprojection.

The forward quadrature matrix and the backprojection gather are deliberately
separate discretizations. Neither geometry is a learned reconstruction operator.
"""
from functools import lru_cache
import itertools
import math
import numpy as np
import torch
from torch import nn
from contextlib import contextmanager
from contextvars import ContextVar

_DENSE_ELEMENT_BUDGET = ContextVar('radon_dense_element_budget', default=8_000_000)


@contextmanager
def geometry_budget(elements):
    """Explicit resource-only bound; legacy callers retain their original limit.

    The quadrature is unchanged. Large-cohort callers must separately pass full
    GPU/host memory admission. At most 128M float64 entries per CPU operator.
    """
    if type(elements) is not int or not 1 <= elements <= 128_000_000:
        raise ValueError('Unsupported dense geometry allocation budget')
    token = _DENSE_ELEMENT_BUDGET.set(elements)
    try:
        yield
    finally:
        _DENSE_ELEMENT_BUDGET.reset(token)

EEM_VERSION = 'antipodal_riesz2_projected_backtracking_v1'


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def householder(n):
    e = np.zeros_like(n); e[0] = 1.
    v = e-n
    if np.linalg.norm(v) < 1e-12:
        return np.eye(len(n), dtype=np.float64)
    v /= np.linalg.norm(v)
    return np.eye(len(n))-2*np.outer(v, v)


@lru_cache(maxsize=64)
def eem_directions(d, M):
    positive_integer(d, 'dimension'); positive_integer(M, 'M')
    if d == 1:
        if M != 1:
            raise ValueError('One-dimensional space has only one unoriented direction; use M=1')
        return np.ones((1, 1)), {'version': EEM_VERSION, 'seeds': [], 'energy': 0., 'iterations': 0}
    seeds = [20260904+100*d+j for j in range(4)]
    if M == 1:
        u = np.zeros((1, d)); u[0, 0] = 1
        return u, {'version': EEM_VERSION, 'seeds': seeds, 'energy': 0., 'iterations': 0}
    pair = np.triu_indices(M, 1)
    def energy(u):
        c = u@u.T
        q = 1-c[pair]**2
        return float(np.sum(1/q)) if np.all(q > 0) else math.inf
    best = None
    for seed in seeds:
        u = np.random.default_rng(seed).normal(size=(M, d))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        value = energy(u)
        for iteration in range(1500):
            c = u@u.T; np.fill_diagonal(c, 0.)
            coefficients = 2*c/np.maximum(1-c*c, 1e-15)**2
            np.fill_diagonal(coefficients, 0.)
            gradient = coefficients@u
            gradient -= np.sum(gradient*u, axis=1, keepdims=True)*u
            norm = np.linalg.norm(gradient)
            if norm < 1e-9:
                break
            step = min(.1, 1/max(norm, 1.))
            for _ in range(30):
                candidate = u-step*gradient
                candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
                candidate_value = energy(candidate)
                if candidate_value < value:
                    u, value = candidate, candidate_value
                    break
                step *= .5
            else:
                break
        if best is None or value < best[0]:
            best = value, u.copy(), iteration+1, seed
    value, u, iterations, chosen_seed = best
    u = u@householder(u[0]).T
    for row in u:
        nonzero = np.flatnonzero(np.abs(row) > 1e-12)
        if row[nonzero[-1]] < 0:
            row *= -1
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    if np.min(1-(u@u.T)[pair]**2) <= 1e-8:
        raise RuntimeError('EEM produced duplicate unoriented directions')
    return u, {'version': EEM_VERSION, 'seeds': seeds, 'chosen_seed': chosen_seed,
               'energy': value, 'iterations': iterations, 'max_iterations': 1500}


def geometry(shape, S, spacing=None):
    shape = tuple(shape); d = len(shape)
    positive_integer(S, 'S', 2)
    if not shape or min(shape) < 2:
        raise ValueError('Every spatial extent must be >=2')
    h = np.asarray(spacing if spacing is not None else (2/max(shape),)*d, dtype=np.float64)
    if h.shape != (d,) or not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ValueError('Invalid spacing')
    radius = float(np.linalg.norm((np.asarray(shape)+1)*h/2))
    s = np.linspace(-radius, radius, S)
    count = max(3, math.ceil(2*radius/h.min())+1)
    t = np.linspace(-radius, radius, count)
    return h, radius, s, t


def raw_operator(shape, M, S, spacing=None):
    # Check before cache lookup: a previous enlarged-budget caller must not make
    # a legacy caller bypass its allocation policy.
    _, _, _, t = geometry(shape, S, spacing)
    if M*S*math.prod(shape) > _DENSE_ELEMENT_BUDGET.get() or S*len(t)**(len(shape)-1) > 2_000_000:
        raise ValueError('Dense reference geometry budget exceeded')
    return _raw_operator_cached(shape, M, S, spacing)


@lru_cache(maxsize=4)
def _raw_operator_cached(shape, M, S, spacing=None):
    directions, _ = eem_directions(len(shape), M)
    h, radius, s, t = geometry(shape, S, spacing)
    d = len(shape); N = math.prod(shape)
    if M*S*N > _DENSE_ELEMENT_BUDGET.get() or S*len(t)**(d-1) > 2_000_000:
        raise ValueError('Dense reference geometry budget exceeded')
    canonical = np.stack(np.meshgrid(s, *([t]*(d-1)), indexing='ij'), -1).reshape(-1, d)
    rows = np.repeat(np.arange(S), len(t)**(d-1))
    quadrature = (t[1]-t[0])**(d-1)
    matrix = np.zeros((M*S, N), dtype=np.float64)
    strides = np.asarray([math.prod(shape[j+1:]) for j in range(d)])
    for m, n in enumerate(directions):
        indices = (canonical@householder(n).T)/h+(np.asarray(shape)-1)/2
        base = np.floor(indices).astype(np.int64); frac = indices-base
        for corner in itertools.product((0, 1), repeat=d):
            corner = np.asarray(corner); ix = base+corner
            valid = ((ix >= 0) & (ix < shape)).all(axis=1)
            w = np.where(corner, frac, 1-frac).prod(axis=1)
            np.add.at(matrix, (m*S+rows[valid], ix[valid]@strides), w[valid]*quadrature)
    return matrix


class Projector(nn.Module):
    def __init__(self, shape, M, S, spacing=None):
        super().__init__()
        positive_integer(M, 'M'); positive_integer(S, 'S', 2)
        self.shape, self.M, self.S = tuple(shape), M, S
        directions, info = eem_directions(len(shape), M)
        h, radius, s, _ = geometry(shape, S, spacing)
        self.metadata = dict(info, directions=directions.tolist(), shape=list(shape), M=M, S=S,
                             support=[-radius, radius], spacing=h.tolist(), angular_weight=math.pi**(len(shape)/2)/math.gamma(len(shape)/2)/M,
                             projection_kind='raw_householder_radon', return_kind='direct_linear_interpolation')
        # Float64 master geometry enables meaningful double precision verification.
        # Training explicitly converts the whole graph to float32.
        self.register_buffer('matrix', torch.from_numpy(raw_operator(tuple(shape), M, S, tuple(h)).copy()))
        coords = np.stack(np.meshgrid(*[(np.arange(n)-(n-1)/2)*step for n, step in zip(shape, h)], indexing='ij'), -1).reshape(-1, len(shape))
        position = (directions@coords.T+radius)/(s[1]-s[0])
        lower = np.floor(position).astype(np.int64); fraction = position-lower
        self.register_buffer('lower', torch.from_numpy(np.clip(lower, 0, S-1)))
        self.register_buffer('upper', torch.from_numpy(np.clip(lower+1, 0, S-1)))
        self.register_buffer('lower_weight', torch.from_numpy((1-fraction)*((lower>=0)&(lower<S))))
        self.register_buffer('upper_weight', torch.from_numpy(fraction*((lower+1>=0)&(lower+1<S))))
        self.angular_weight = self.metadata['angular_weight']

    def forward(self, x):
        if tuple(x.shape[2:]) != self.shape:
            raise ValueError('Native feature shape changed')
        p = x.flatten(2)@self.matrix.T
        return p.reshape(x.shape[0], x.shape[1]*self.M, self.S)

    def backproject(self, p):
        b, cm, s = p.shape
        if s != self.S or cm % self.M:
            raise ValueError('Projection shape mismatch')
        curves = p.reshape(b, cm//self.M, self.M, self.S)
        delta = p.new_zeros((b, cm//self.M, math.prod(self.shape)))
        for m in range(self.M):
            curve = curves[:, :, m]
            delta = delta+curve[..., self.lower[m]]*self.lower_weight[m]+curve[..., self.upper[m]]*self.upper_weight[m]
        return (delta*self.angular_weight).reshape(b, cm//self.M, *self.shape)


class ScrambledProjector(Projector):
    """Destroy spatial arrangement with an invertible fixed voxel permutation."""
    def __init__(self, shape, M, S, seed):
        super().__init__(shape, M, S)
        permutation=np.random.default_rng(seed).permutation(math.prod(shape))
        self.register_buffer('permutation',torch.from_numpy(permutation))
        self.register_buffer('inverse_permutation',torch.from_numpy(np.argsort(permutation)))
        self.metadata.update(projection_kind='spatial_permutation_radon',random_seed=seed,
                             return_kind='inverse_permutation_direct_bp')
    def forward(self,x):
        permuted=x.flatten(2)[...,self.permutation].reshape_as(x)
        return super().forward(permuted)
    def backproject(self,p):
        x=super().backproject(p)
        return x.flatten(2)[...,self.inverse_permutation].reshape_as(x)


class GaussianProjector(Projector):
    """Fixed Gaussian linear control; norm-matched forward and scaled adjoint return.

    This is an explicitly defined experimental control, not a claim to reproduce
    a separately published method named Random Bridge. It has no learned geometry.
    """
    def __init__(self,shape,M,S,seed):
        super().__init__(shape,M,S)
        rng=np.random.default_rng(seed)
        q=rng.normal(size=tuple(self.matrix.shape))
        q*=np.linalg.norm(self.matrix.numpy(),axis=1,keepdims=True)/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-300)
        # Ordinary BP matrix, used only to match row energy of the random return.
        N=math.prod(shape); reference=np.zeros((N,M*S))
        rows=np.arange(N)
        for m in range(M):
            np.add.at(reference,(rows,m*S+self.lower[m].numpy()),self.lower_weight[m].numpy()*self.angular_weight)
            np.add.at(reference,(rows,m*S+self.upper[m].numpy()),self.upper_weight[m].numpy()*self.angular_weight)
        back=q.T.copy()
        back*=np.linalg.norm(reference,axis=1,keepdims=True)/np.maximum(np.linalg.norm(back,axis=1,keepdims=True),1e-300)
        self.matrix.copy_(torch.from_numpy(q))
        self.register_buffer('return_matrix',torch.from_numpy(back))
        self.metadata.update(projection_kind='row_norm_matched_fixed_gaussian',random_seed=seed,
                             return_kind='row_norm_matched_scaled_adjoint',
                             matching='forward row norms and return row norms; singular values are not matched')
    def backproject(self,p):
        b,cm,_=p.shape
        x=p.reshape(b,cm//self.M,self.M*self.S)@self.return_matrix.T
        return x.reshape(b,cm//self.M,*self.shape)


class LinearResampleProjector(Projector):
    """Flattened linear interpolation, matched to Radon/BP row norms separately."""
    def __init__(self, shape, M, S):
        super().__init__(shape, M, S)
        n=math.prod(shape); t=M*S
        # Each input identity row is one channel. Interpolate only the last axis.
        forward=torch.nn.functional.interpolate(torch.eye(n,dtype=torch.float64)[None],size=t,mode='linear',align_corners=False)[0].T.contiguous()
        backward=torch.nn.functional.interpolate(torch.eye(t,dtype=torch.float64)[None],size=n,mode='linear',align_corners=False)[0].T.contiguous()
        reference=Projector.backproject(self,torch.eye(t,dtype=torch.float64).reshape(t,M,S)).reshape(t,n).T
        for matrix,target in [(forward,self.matrix),(backward,reference)]:
            norms=matrix.norm(dim=1,keepdim=True);target_norms=target.norm(dim=1,keepdim=True)
            matrix.mul_(torch.where(norms>0,target_norms/norms.clamp_min(1e-300),torch.zeros_like(norms)))
        self.matrix.copy_(forward);self.register_buffer('return_matrix',backward)
        self.metadata.update(projection_kind='row_norm_matched_flat_linear_resampling',return_kind='row_norm_matched_flat_linear_resampling_return',
                             align_corners=False,spatial_order='native contiguous flatten',packing_factor=M,angles_applicable=False,
                             matching='forward/return row L2 norms and mixer width; rank and singular values not matched')
    def backproject(self,p):
        b,cm,s=p.shape
        if s!=self.S or cm%self.M:raise ValueError('Resampling packed shape mismatch')
        return (p.reshape(b,cm//self.M,self.M*self.S)@self.return_matrix.T).reshape(b,cm//self.M,*self.shape)
