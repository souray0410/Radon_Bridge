"""Checkpoint provenance, control isolation, and modality-independent stage one."""
import argparse,json,tempfile,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from radonbridge.bridge import BridgeExchange,FeatureSpec
from radonbridge.projector import Projector,GaussianProjector,ScrambledProjector
from radonbridge.model import PilotGraph
from radonbridge.experiment import bridge_configs,parameter_hash
from radonbridge.optimization import configure_optimizer,clip_task_gradients


def control_checks():
    shape=(3,4);M,S=4,9
    p=Projector(shape,M,S);q=GaussianProjector(shape,M,S,42);r=GaussianProjector(shape,M,S,42)
    assert torch.equal(q.matrix,r.matrix)
    assert torch.allclose(q.matrix.norm(dim=1),p.matrix.norm(dim=1),atol=1e-12,rtol=1e-12)
    bp=p.backproject(torch.eye(M*S,dtype=torch.float64).reshape(M*S,M,S)).flatten(1).T
    assert torch.allclose(q.return_matrix.norm(dim=1),bp.norm(dim=1),atol=1e-12,rtol=1e-12)
    scrambled=ScrambledProjector(shape,M,S,43)
    assert torch.equal(scrambled.matrix,p.matrix)
    x=torch.randn(2,3,*shape,dtype=torch.float64)
    assert torch.equal(scrambled(x),p(x.flatten(2)[...,scrambled.permutation].reshape_as(x)))
    z=torch.randn(2,3*M,S,dtype=torch.float64)
    assert torch.equal(scrambled.backproject(z),p.backproject(z).flatten(2)[...,scrambled.inverse_permutation].reshape_as(x))
    specs=[FeatureSpec('a',2,(3,4)),FeatureSpec('b',3,(3,4,3))]
    # Heterogeneous participant widths and a one-channel boundary never expand.
    for ratio in [.001,.125,.5,1.]:
        module=BridgeExchange(specs,M=4,S=9,rho=ratio)
        for spec,layer in zip(specs,module.compress):
            assert layer.out_channels==max(1,int(ratio*spec.channels*4))
            assert 1<=layer.out_channels<=layer.in_channels
    for ratio in [0,-.1,1.1,float('nan'),float('inf'),True]:
        try:BridgeExchange(specs,M=4,S=9,rho=ratio)
        except ValueError:pass
        else:raise AssertionError('Invalid ratio accepted')
    states=[];counts=[]
    for mode in ['radon','random','scrambled']:
        torch.manual_seed(117)
        module=BridgeExchange(specs,M=4,S=9,rho=.5,mode=mode).double()
        states.append({k:v.detach().clone() for k,v in module.named_parameters()});counts.append(sum(p.numel() for p in module.parameters()))
        assert all(not b.requires_grad for b in module.buffers())
        nn.init.normal_(module.mixer.conv.weight,std=.01)
        inputs=[torch.randn(2,s.channels,*s.shape,dtype=torch.float64,requires_grad=True) for s in specs]
        module(*inputs).square().mean().backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in inputs)
    assert len(set(counts))==1
    assert all(all(torch.equal(states[0][k],s[k]) for k in states[0]) for s in states[1:])
    return {'fixed_random_reproducible':True,'random_forward_and_return_row_norms_matched':True,
            'scramble_forward_inverse_verified':True,'learnable_initialization_and_parameter_count_identical':counts[0]}


def checkpoint_checks(device):
    torch.manual_seed(123)
    g=PilotGraph(seed=123,device=device)
    c=torch.randn(2,2,3,224,224,device=device);o=torch.randn(2,2,1,32,96,96,device=device);y=torch.tensor([0,1],device=device)
    recipe={'adapt_stages':[1,2,3,4],'backbone_lr':3e-5,'head_bridge_lr':1e-4,'weight_decay':.01,'training_regime':'full_finetune'}
    opt=configure_optimizer(g,recipe)
    g.graph.train();g.forward(c,o,y)
    # No communication: a CFP loss has no path to OCT parameters.
    other=next(g.modules_by_name()['oct_stage1'].parameters())
    assert torch.autograd.grad(g.by_name['cfp_loss'].feature_message.current_state,other,allow_unused=True,retain_graph=True)[0] is None
    g.backward();clip_task_gradients(g,5.);opt.step()
    g.graph.eval();expected,_=g.forward(c,o,y);expected={k:v.detach().clone() for k,v in expected.items()}
    state=g.save_state();digest=parameter_hash(g)
    del opt,g
    if device=='cuda':torch.cuda.empty_cache()
    verified=[]
    for mode in ['independent','radon','self','pooled','random','scrambled']:
        loaded=PilotGraph(seed=123,device=device,bridge_configs=[] if mode=='independent' else bridge_configs((3,),mode))
        for branch in ['cfp','oct']:loaded.load_native_state({k:v for k,v in state.items() if k.startswith(branch+'_')},branch)
        assert parameter_hash(loaded)==digest
        opt=configure_optimizer(loaded,recipe);assert not opt.state
        loaded.graph.eval();actual,_=loaded.forward(c,o,y)
        assert all(torch.equal(actual[k],expected[k]) for k in expected),mode
        assert all(p.requires_grad for p in loaded.graph.parameters())
        verified.append(mode);del opt,loaded
        if device=='cuda':torch.cuda.empty_cache()
    return {'stage_one_has_no_cross_modality_gradient':True,'strict_native_checkpoint_load':True,
            'all_six_arms_initial_predictions_exactly_identical':verified,'optimizer_fresh_all_arms':True,'all_parameters_unfrozen':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(3)
    result={'controls':control_checks(),'checkpoints':checkpoint_checks(a.device),'passed':True}
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
