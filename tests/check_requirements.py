"""Specification-driven complete-group and separate-predictor acceptance checks."""
import json
import torch
import numpy as np
from torch import nn
from radonbridge.bridge import FeatureSpec,attach_group,LinearMixer
from radonbridge.graph import MHDBuilder
from radonbridge.model import PilotGraph
from radonbridge.metrics import classification_metrics

class SquareLoss(nn.Module):
    def forward(self,*features):return sum(x.square().mean() for x in features)


def exercise(specs,repeats=1,upsilon=(1.,1.,.25),mode="radon",device="cpu"):
    b=MHDBuilder();inputs={s.key:b.node(s.key) for s in specs};outputs=inputs;metas=[]
    for j in range(repeats):outputs,meta=attach_group(b.node,b.edge,specs,outputs,f"group{j}_",upsilon,mode);metas.append(meta)
    loss=b.node("loss");b.edge("objective",SquareLoss(),list(outputs.values()),[loss]);g=b.compile(device)
    x={s.key:torch.randn((2,s.channels)+s.shape,device=device,requires_grad=True) for s in specs}
    b.set_inputs(x);g.forward(levels=b.forward_levels)
    for k,n in outputs.items():assert torch.equal(b.nodes[n].feature_message.current_state,x[k])
    for e in b.edges:
        m=e.edge_operations[0].function
        if isinstance(m,LinearMixer):nn.init.normal_(m.conv.weight,std=.02)
    g.zero_grad(set_to_none=True);b.set_inputs(x);g.forward(levels=b.forward_levels)
    first=b.nodes[outputs[specs[0].key]].feature_message.current_state
    cross=torch.autograd.grad(first.square().mean(),x[specs[-1].key],allow_unused=True,retain_graph=True)[0]
    if mode=="self":assert cross is None or torch.count_nonzero(cross)==0
    elif len(specs)>1:assert cross is not None and cross.abs().sum()>0
    g.backward(levels=b.backward_levels)
    grads=[None if p.grad is None else p.grad.clone() for p in g.parameters()]
    for v in x.values():v.grad=None
    g.zero_grad(set_to_none=True);b.native_forward(x)["loss"].backward()
    for p,grad in zip(g.parameters(),grads):
        assert (p.grad is None)==(grad is None)
        if grad is not None:assert torch.allclose(p.grad,grad,atol=2e-6,rtol=1e-4)
    return metas


def main():
    torch.set_num_threads(2);torch.manual_seed(901)
    def spec(k,d,c):return FeatureSpec(k,c,tuple(3+i%2 for i in range(d)),(3,)*(d-1))
    cases=[(2,3),(2,2),(3,3),(1,2,3),(1,2,3,4)]
    for dims in cases:exercise([spec(f"p{i}",d,i+2) for i,d in enumerate(dims)],repeats=2)
    specs=[spec("a",2,2),spec("b",3,3)]
    base=exercise(specs)[0];m=exercise(specs,upsilon=(.5,1.,.25))[0];s=exercise(specs,upsilon=(1.,.5,.25))[0];h=exercise(specs,upsilon=(1.,1.,.5))[0]
    assert base["span"]==m["span"]==h["span"] and s["span"]<base["span"]
    for i in range(2):
        a=base["participants"][i]
        assert a["support"]==s["participants"][i]["support"] and a["mesh"]==s["participants"][i]["mesh"]
        assert a["P"]==h["participants"][i]["P"] and a["H"]<h["participants"][i]["H"]
    exercise(specs,mode="self");exercise(specs,upsilon=(1.,.01,.25))
    exercise([FeatureSpec("x",2,(3,4),(3,),coordinate_mode="physical",spacing=(.8,1.2)),spec("y",1,3)])
    # Complete independent task graphs with a true no-cross-information baseline.
    c=torch.randn(1,2,3,96,96);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([1])
    baseline=PilotGraph();bl,_=baseline.forward(c,o,y)
    for branch in ("cfp","oct"):
        mono=PilotGraph(modalities=branch);z,_=mono.forward(c,o,y);assert torch.equal(bl[branch],z[branch])
    changed,_=baseline.forward(c,o+torch.randn_like(o),y);assert torch.equal(bl["cfp"],changed["cfp"])
    for stages in ((2,),(3,),(2,3)):
        g=PilotGraph("radon",bridge_stages=stages,handoff_ratio=.03125);z,l=g.forward(c,o,y)
        assert set(z)=={"cfp","oct"} and "joined" not in g.by_name and "classifier" not in g.modules_by_name()
        assert all(torch.equal(z[k],bl[k]) for k in z)
        g.backward()
        assert all(g.modules_by_name()[k+"_head"].weight.grad.abs().sum()>0 for k in z)
        expected=sum(g.by_name[k+"_loss"].feature_message.current_state for k in z)/2
        assert torch.allclose(l,expected)
    q=classification_metrics([0,0,1,1],np.array([[1.,0.]]*4));assert abs(q["macro_f1"]-1/3)<1e-12
    gpu_checked=False
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
        exercise(specs,device="cuda");gpu_checked=True
        from radonbridge.projector import Projector
        p=Projector((4,5),(4,),9);x=torch.randn(2,3,4,5)
        assert torch.allclose(p(x),p.cuda()(x.cuda()).cpu(),atol=1e-5,rtol=1e-4)
    print(json.dumps({"complete_groups":cases,"repeated_groups":2,"cross_gradients":True,"self_blocks_cross_gradients":True,"mhd_native_gradients_match":True,"separate_heads":True,"standalone_equals_no_bridge":True,"locations":[[2],[3],[2,3]],"msh_controls":True,"physical_spacing":True,"macro_f1_verified":True,"gpu_checked":gpu_checked}))

if __name__=="__main__":main()
