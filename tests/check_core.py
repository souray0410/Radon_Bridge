"""Meaningful numerical acceptance tests for the small-feature pilot."""
import json
import time
import torch
import numpy as np
from radonbridge.projector import Projector, orientations, householder
from radonbridge.model import PilotGraph


def main():
    torch.set_num_threads(2);torch.manual_seed(1)
    report={};start=time.time()
    for d in range(1,5):
        mesh=(3,)*(d-1); ns,w=orientations(mesh)
        herr=max(np.max(np.abs(householder(n) @ householder(n)-np.eye(d))) for n in ns)
        nerr=max(np.linalg.norm(householder(n)[:,0]-n) for n in ns)
        assert herr<1e-12 and nerr<1e-12
        p=Projector((4,)*d,mesh,9).double()
        x=torch.randn((1,2)+p.shape,dtype=torch.double,requires_grad=True)
        z=torch.randn(1,2*p.directions,9,dtype=torch.double,requires_grad=True)
        lhs=(p(x)*z).sum();rhs=(x*p.adjoint(z)).sum()
        error=float((lhs-rhs).abs().detach())
        assert error<1e-10
        assert torch.autograd.gradcheck(p,(x,),fast_mode=True)
        assert torch.autograd.gradcheck(p.adjoint,(z,),fast_mode=True)
        assert (p.matrix>=0).all()
        report[f"{d}d"]={"householder_error":herr,"mapping_error":nerr,"adjoint_abs_error":error,"gradient_check":True}
    c=torch.randn(1,2,3,96,96);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([1])
    baseline=PilotGraph();b_logits,_=baseline.forward(c,o,y)
    g=PilotGraph("radon"); logits,loss=g.forward(c,o,y)
    assert torch.equal(logits,b_logits),"Zero bridge must preserve the native model"
    # A nonzero mixer is needed to exercise both source-to-target gradients.
    torch.nn.init.normal_(g.modules_by_name()["projection_mixer"].conv.weight,std=.002)
    g.graph.zero_grad(set_to_none=True)
    logits,loss=g.forward(c,o,y);g.backward()
    parameters=list(g.graph.parameters())
    grad=[None if p.grad is None else p.grad.clone() for p in parameters]
    assert g.by_name["cfp_stage3"].gradient_message.current_state.abs().sum()>0
    assert g.by_name["oct_stage3"].gradient_message.current_state.abs().sum()>0
    lr=.001
    expected=[p.detach().clone() if v is None else p.detach()-lr*v for p,v in zip(parameters,grad)]
    g.graph.zero_grad(set_to_none=True)
    nlog,nloss=g.native_forward(c,o,y);nloss.backward()
    assert torch.allclose(logits,nlog,atol=1e-6)
    errors=[]
    for p,v in zip(parameters,grad):
        assert (p.grad is None)==(v is None)
        if v is not None:
            errors.append(float((p.grad-v).abs().max()))
            assert torch.allclose(p.grad,v,rtol=1e-4,atol=2e-6)
    torch.optim.SGD(parameters,lr=lr).step()
    assert all(torch.allclose(p,e,atol=2e-6) for p,e in zip(parameters,expected))
    report["mhd"]={"nodes":len(g.nodes),"edges":len(g.edges),"max_gradient_error":max(errors),"update_matches":True,"zero_bridge_exact":True}
    report["seconds"]=time.time()-start
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
