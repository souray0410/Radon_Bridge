import numpy as np
import torch
from radon_bridge.methods.projection import Projector, eem_directions
from radon_bridge.methods.operator import LinearMixer

def test_fixed_geometry_linear_and_zero_mixer():
    p=Projector((3,4),4,11).double()
    x=torch.randn(2,3,3,4,dtype=torch.float64);y=torch.randn_like(x)
    assert torch.allclose(p(x+y),p(x)+p(y),atol=1e-11,rtol=1e-11)
    mixer=LinearMixer([4,4],3).double()
    a,b=mixer(torch.randn(2,4,11,dtype=torch.float64),torch.randn(2,4,11,dtype=torch.float64))
    assert torch.count_nonzero(a)==torch.count_nonzero(b)==0

def test_eem_directions_unit_norm():
    for d in (2,3):
        q,_=eem_directions(d,8)
        assert np.allclose(np.linalg.norm(q,axis=1),1)
