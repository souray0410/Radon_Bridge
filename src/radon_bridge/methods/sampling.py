"""Prespecified common S-axis reindexing; never consumes training RNG state."""
import hashlib
import json
import numpy as np

VERSION = 'radon_bridge_s_axis_adjacency_v1'


def validate_permutation(values, S=None):
    if not isinstance(values, (list, tuple)) or len(values)<2:
        raise ValueError('S permutation must be a nonempty integer list of length >=2')
    if any(type(v) is not int for v in values) or sorted(values)!=list(range(len(values))):
        raise ValueError('S permutation must be an exact bijection')
    if S is not None and len(values)!=S:raise ValueError('S permutation length mismatch')
    return list(values)


def permutation_metadata(values):
    values=validate_permutation(values)
    payload=json.dumps(values,separators=(',',':')).encode()
    return dict(permutation=values,sha256=hashlib.sha256(payload).hexdigest(),
                sha_encoding='UTF-8 compact JSON integer array',
                convention='z[..., permutation] -> Conv1d -> y[..., argsort(permutation)]',
                shared_across='all sources, channels, training seeds, bases and rho',
                boundary_note='Zero-padding boundary locations are permuted along with adjacency; not isolated from boundary effects.')


def fixed_permutation(S=64):
    key=f'{VERSION}|S={S}'
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')
    values=np.random.Generator(np.random.PCG64(seed)).permutation(S).tolist()
    return dict(permutation_metadata(values),version=VERSION,seed=seed,seed_key=key,
                generator='NumPy PCG64; independent generator; no rejection sampling',
                original_adjacent_pairs_remaining=sum(abs(a-b)==1 for a,b in zip(values[:-1],values[1:])))
