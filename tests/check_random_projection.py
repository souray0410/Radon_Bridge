"""Pending ws acceptance for the fixed random linear projection control."""
import json
import torch
from radonbridge.projector import Projector
from radonbridge.model import PilotGraph
from check_requirements import exercise
from radonbridge.bridge import FeatureSpec

def main():
    torch.set_num_threads(2);torch.manual_seed(470)
    rows=[]
    for shape,mesh in [((14,14),(8,)),((8,6,6),(4,4))]:
        radon=Projector(shape,mesh,23);random=Projector(shape,mesh,23,random_projection=True)
        repeat=Projector(shape,mesh,23,random_projection=True)
        assert torch.equal(random.matrix,repeat.matrix) and not torch.equal(radon.matrix,random.matrix)
        assert torch.allclose(radon.matrix.norm(dim=1),random.matrix.norm(dim=1),atol=1e-7,rtol=1e-5)
        random=random.double();x=torch.randn((2,3)+shape,dtype=torch.double);z=torch.randn(2,3*random.directions,23,dtype=torch.double)
        assert torch.allclose((random(x)*z).sum(),(x*random.adjoint(z)).sum(),atol=1e-10,rtol=1e-10)
        rows.append({'reference':radon.diagnostics(),'random':random.diagnostics()})
    specs=[FeatureSpec('a',2,(4,5),(3,)),FeatureSpec('b',3,(3,4,5),(3,3))]
    exercise(specs,repeats=2,mode='random')
    baseline=PilotGraph(loss_reduction='sum');random=PilotGraph('random',bridge_stages=(3,),upsilon=(1.,1.,1/32),loss_reduction='sum')
    radon=PilotGraph('radon',bridge_stages=(3,),upsilon=(1.,1.,1/32),loss_reduction='sum')
    assert sum(p.numel() for p in random.graph.parameters())==sum(p.numel() for p in radon.graph.parameters())
    c=torch.randn(1,2,3,96,96);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([0])
    a,_=baseline.forward(c,o,y);b,_=random.forward(c,o,y)
    assert all(torch.equal(a[k],b[k]) for k in a)
    gpu=False
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
        exercise(specs,mode='random',device='cuda');gpu=True
    print(json.dumps({'deterministic_fixed_draw':True,'row_norms_match':True,'adjoint':True,'zero_bridge_exact':True,
                      'equal_learned_parameter_count':True,'complete_mhd_gradient_checks':True,'gpu_checked':gpu,'spectral_diagnostics':rows}))
if __name__=='__main__':main()
