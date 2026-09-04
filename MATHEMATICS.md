# Mathematical contract for the pilot

For native feature `F[B,C,N1,...,Nd]`, the half-sphere orientation mesh uses
`theta_j = pi*m/M_j` and signed distance. The standard hyperspherical mapping
produces unit direction n. With `v=(e1-n)/||e1-n||`, `H=I-2vv^T` maps e1 to n.
For n=e1 use identity. Direction degeneracies are permitted; angular parameter
cell integrals of the sphere measure provide normalized orientation weights.

For each direction, canonical coordinates `(s,t2,...,td)` are mapped by H into
the native lattice. Zero-extended multilinear interpolation is integrated over
the transverse coordinates. This constructs matrix A. The distance and
transverse ranges include the full support of the interpolation basis, not
merely the native center coordinates. 1D has no transverse integral or angles.

Native spacing is normalized from feature shape; it is not a claim of physical
CFP/OCT registration. Quadrature and sqrt angular weights enter A. A fixed
normalizer `sqrt(||A||_1 ||A||_infinity)` bounds its Euclidean operator norm.
This geometry-only normalization is recorded and is not a learned residual gate.

The return operator is exactly the transpose of this discrete A. It is not an
inverse Radon transform. Tests verify `<Ax,z>=<x,A^Tz>` and gradients in both
directions. Householder's continuous self-inverse property does not establish
that repeated interpolation is self-inverse.

Flattening channels and directions produces `[B,P,S]`, where P=C*prod(M).
Post-projection linear compression uses configurable H=round(upsilon_H*P); the
small-cohort default is upsilon_H=0.03125 after the initial 0.25 pilot overfit. Compressed tensors are
concatenated, linearly convolved along S, split by participant branch, expanded,
backprojected and directly added to their own native feature. Thus the fixed-
geometry bridge is one structured linear map. Geometry's advantage is an
experimental hypothesis, not a consequence of matching tensor shapes.

The scrambled control permutes A's native columns with a fixed seed. It retains
the same singular spectrum, norm bound, dimensions and compute while destroying
native spatial adjacency. This specifically tests geometry; it is not a full
substitute for random or learned projection baselines.
