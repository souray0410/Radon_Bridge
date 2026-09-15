"""Source-resolved diagnostics of W[t] = B K[t] A, without dense W export.

The source blocks are columns of A and destination blocks are rows of B.
A latent block of K is not a source edge. This module never changes a weight,
mask, parameter gradient or training policy; overrides describe inference only.
"""
from itertools import accumulate
import torch
from torch.nn import functional as F
from radon_bridge.methods.factorized import FactorizedMixer


def source_messages(mixer, features, overrides=None):
    """Return each destination's sum; override (destination, source) with Z or None.

Z has the original source projected width, not the global bottleneck rank.
Self terms must remain unchanged. Deletion returns a functional intervention at
fixed weights; it does not implement a self-only or unidirectional training arm.
"""
    if not isinstance(mixer, FactorizedMixer):
        raise TypeError('Source decomposition requires FactorizedMixer')
    widths=mixer.widths; n=len(widths); offsets=(0,*accumulate(widths))
    if len(features)!=n or not n:
        raise ValueError('Source count differs from the mixer contract')
    weight=mixer.encoder.weight
    for x,w in zip(features,widths):
        if (not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[1]!=w or
                x.shape[0]!=features[0].shape[0] or x.shape[2]!=features[0].shape[2] or
                x.dtype!=weight.dtype or x.device!=weight.device):
            raise ValueError('Expected aligned [observed eyes, source CM, S] tensors')
    for layer in (mixer.encoder,mixer.decoder,mixer.conv):
        if (layer.bias is not None or layer.groups!=1 or layer.stride!=(1,) or
                layer.dilation!=(1,) or layer.padding_mode!='zeros'):
            raise ValueError('Unsupported factorized operator semantics')
    if (mixer.encoder.kernel_size!=(1,) or mixer.decoder.kernel_size!=(1,) or
            mixer.encoder.padding!=(0,) or mixer.decoder.padding!=(0,) or
            mixer.conv.padding!=(mixer.conv.kernel_size[0]//2,)):
        raise ValueError('Unexpected factorized kernel or padding')
    if not torch.all(mixer.mask==1):
        raise ValueError('A latent mask is not an accepted source-edge definition')
    overrides={} if overrides is None else overrides
    for edge,x in overrides.items():
        if (not isinstance(edge,tuple) or len(edge)!=2 or
                any(type(i) is not int or not 0<=i<n for i in edge) or edge[0]==edge[1]):
            raise ValueError('Only declared cross-source edges may be overridden')
        if x is not None:
            original=features[edge[1]]
            if (not isinstance(x,torch.Tensor) or x.shape!=original.shape or
                    x.dtype!=original.dtype or x.device!=original.device):
                raise ValueError('Donor must preserve source width, eye count and sampling')
    # K is shared. Cache K(A_j Z_j) once per original source, never construct D²k.
    def inner(src,x):
        projected=F.conv1d(x,weight[:,offsets[src]:offsets[src+1],:])
        return F.conv1d(projected,mixer.conv.weight,padding=mixer.conv.padding)
    originals=[inner(j,x) for j,x in enumerate(features)]
    outputs=[]
    for dst in range(n):
        terms=[]
        for src in range(n):
            if (dst,src) not in overrides:terms.append(originals[src])
            elif overrides[(dst,src)] is not None:terms.append(inner(src,overrides[(dst,src)]))
        total=terms[0]
        for term in terms[1:]:total=total+term
        outputs.append(F.conv1d(total,mixer.decoder.weight[offsets[dst]:offsets[dst+1],:,:]))
    return tuple(outputs)
