"""Independent quadrature/warp references plus MHD topology and gradient checks."""
import argparse
import json
import math
import time
import numpy as np
import torch
from torch import nn
from scipy.ndimage import map_coordinates
from radonbridge.projector import Projector, eem_directions, geometry, householder
from radonbridge.bridge import attach_to_nodes, LinearMixer
from radonbridge.graph import MHDBuilder


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12))


def geometry_check(d, device):
    shape = tuple(3+j%2 for j in range(d)); M, S = 4, 11
    p = Projector(shape, M, S)
    h, r, s, t = geometry(shape, S)
    ns, _ = eem_directions(d, M)
    rng = np.random.default_rng(124+d)
    x = rng.normal(size=shape)
    # Reference materializes each full transformed grid and integrates it.
    yy = np.stack(np.meshgrid(s, *([t]*(d-1)), indexing='ij'), -1)
    forward = []
    for n in ns:
        xx = yy@householder(n).T
        indices = xx/h+(np.asarray(shape)-1)/2
        rotated = map_coordinates(x, np.moveaxis(indices, -1, 0), order=1, mode='grid-constant', cval=0., prefilter=False)
        forward.append(rotated.sum(axis=tuple(range(1,d)))*(t[1]-t[0])**(d-1))
    forward = np.asarray(forward)
    dtype = torch.float64 if device == 'cpu' else torch.float32
    p = p.to(device=device, dtype=dtype)
    got = p(torch.tensor(x, dtype=dtype, device=device)[None,None]).detach().cpu().numpy().reshape(M,S)
    fe = relative(got, forward)
    curves = rng.normal(size=(M,S))
    native = np.stack(np.meshgrid(*[(np.arange(n)-(n-1)/2)*step for n,step in zip(shape,h)], indexing='ij'), -1)
    reference = np.zeros(shape)
    for n, g in zip(ns, curves):
        expanded = np.broadcast_to(g.reshape(S,*([1]*(d-1))), (S,*([len(t)]*(d-1))))
        inverse = native@np.linalg.inv(householder(n)).T
        indices = inverse.copy(); indices[...,0]=(inverse[...,0]-s[0])/(s[1]-s[0])
        indices[...,1:] = (inverse[...,1:]-t[0])/(t[1]-t[0])
        reference += map_coordinates(expanded, np.moveaxis(indices,-1,0), order=1, mode='grid-constant', cval=0., prefilter=False)
    reference *= math.pi**(d/2)/math.gamma(d/2)/M
    z = torch.tensor(curves.reshape(1,M,S), dtype=dtype, device=device, requires_grad=True)
    back = p.backproject(z)
    be = relative(back.detach().cpu().numpy()[0,0], reference)
    back.square().sum().backward(); assert torch.isfinite(z.grad).all()
    threshold = 1e-10 if device == 'cpu' else 1e-4
    assert fe < threshold and be < threshold, (d, device, fe, be)
    # Separate raw forward and direct BP need not be an exact discrete adjoint pair.
    return {'dimension':d, 'device':device, 'forward_relative_error':fe, 'direct_bp_relative_error':be}


class Stage(nn.Module):
    def __init__(self):
        super().__init__(); self.weight=nn.Parameter(torch.tensor(1.1))
    def forward(self,x): return torch.tanh(self.weight*x)

class Objective(nn.Module):
    def forward(self,*xs): return sum(x.square().mean() for x in xs)


def topology_check(dims, repeated=False, participant_sampling=False, compression="learned_projected", basis_files=None, mode="radon"):
    b=MHDBuilder(); inputs={}; cuts=[]; later=[]; outputs=[]
    for i,d in enumerate(dims):
        key=f'network{i}'; previous=b.node(key)
        inputs[key]=torch.randn((2,i+2)+tuple(3+j%2 for j in range(d)), dtype=torch.float64, requires_grad=True)
        for depth in range(1,4):
            name=f'{key}_stage{depth}'; out=b.node(name); b.edge(name,Stage(),[previous],[out]); previous=out
            if depth == 1+i%2: cuts.append(name)
            if depth == 3: later.append(name)
        outputs.append(previous)
    loss=b.node('loss'); b.edge('objective',Objective(),outputs,[loss])
    original_nodes={k:n.id for k,n in b.by_name.items()}; original_edges=list(b.steps)
    samples=b.native_forward(inputs); baseline=samples['loss'].detach()
    directions={key:4*(i+1) for i,key in enumerate(cuts)} if participant_sampling else 4
    ratios={key:.5/(i+1) for i,key in enumerate(cuts)} if participant_sampling else .5
    _, meta=attach_to_nodes(b,cuts,prefix='bridge_0_',M=directions,S=11,rho=ratios,samples=samples,compression=compression,basis_files=basis_files,mode=mode)
    if participant_sampling:
        assert [x['retained_channels'] for x in meta['participants']] == [2*(i+2) for i in range(len(cuts))]
    if repeated:
        _, second=attach_to_nodes(b,later,prefix='bridge_1_',M=3,S=9,rho=.25,samples=samples)
        assert (meta['M'],meta['S'],meta['rho']) != (second['M'],second['S'],second['rho'])
    g=b.compile('cpu').double(); b.set_inputs(inputs); g.forward(levels=b.forward_levels)
    assert torch.equal(b.by_name['loss'].feature_message.current_state, baseline)
    assert all(b.by_name[k].id==v for k,v in original_nodes.items())
    assert {e:(h,t) for e,h,t in b.steps if e<len(original_edges)} == {e:(h,t) for e,h,t in original_edges}
    for prefix in ['bridge_0_']+(['bridge_1_'] if repeated else []):
        ex=next(e.id for e in b.edges if e.name==prefix+'exchange')
        ret=[e.id for e in b.edges if e.name.startswith(prefix) and e.name.endswith('_return')]
        assert {b.forward_edge_levels[e] for e in ret} == {b.forward_edge_levels[ex]+1}
    for m in g.modules():
        if isinstance(m,LinearMixer): nn.init.normal_(m.conv.weight,std=.02)
    g.zero_grad(set_to_none=True); b.set_inputs(inputs); g.forward(levels=b.forward_levels); g.backward(levels=b.backward_levels)
    saved=[p.grad.clone() for p in g.parameters()]
    g.zero_grad(set_to_none=True); b.native_forward(inputs)['loss'].backward()
    error=max(relative(a.numpy(),p.grad.numpy()) for a,p in zip(saved,g.parameters()))
    assert error<1e-10
    values=b.native_forward(inputs)
    grad=torch.autograd.grad(values[later[0]].square().mean(),inputs['network1'])[0]
    assert (grad.abs().sum()==0) if mode=='self' else (grad.abs().sum()>0)
    # Bridge itself is linear, including its residual return.
    exchange=next(e.edge_operations[0].function for e in b.edges if e.name=='bridge_0_exchange')
    x=[samples[k].detach() for k in cuts]; y=[torch.randn_like(v) for v in x]
    linear=relative(exchange(*[a+c for a,c in zip(x,y)]).detach().numpy(),(exchange(*x)+exchange(*y)).detach().numpy())
    assert linear<1e-10
    return {'dimensions':dims,'repeated_distinct_configs':repeated,'participant_sampling':participant_sampling,'native_topology_preserved':True,
            'zero_bridge_exact':True,'mhd_gradient_relative_error':error,'cross_branch_gradient':float(grad.norm()),'linearity_error':linear}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--device',default='cpu'); parser.add_argument('--output',required=True)
    a=parser.parse_args(); torch.set_num_threads(3); torch.manual_seed(777)
    start=time.monotonic(); result={}
    if a.device=='cpu':
        errors={}
        for M in [2,3,4,8,16]:
            u,_=eem_directions(2,M); angles=np.sort(np.mod(np.arctan2(u[:,1],u[:,0]),np.pi))
            error=float(np.max(np.abs(np.diff(np.r_[angles,angles[0]+np.pi])-np.pi/M))*180/np.pi)
            assert error<1e-4,(M,error); errors[M]=error
        result['eem_2d_gap_degrees']=errors
        result['topology']=[topology_check(dims,repeat) for dims,repeat in [([2,3],False),([2,3,4],False),([2,2,3,4],True)]]
        result['topology'].append(topology_check([2,3],participant_sampling=True))
    result['geometry']=[geometry_check(d,a.device) for d in [2,3,4]]
    result['seconds']=time.monotonic()-start; result['passed']=True
    from pathlib import Path
    Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)

if __name__=='__main__':main()
