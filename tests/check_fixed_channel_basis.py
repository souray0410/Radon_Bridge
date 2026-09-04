"""Fixed channel-basis reproducibility, linearity, topology and optimizer acceptance."""
import json
from pathlib import Path
import tempfile
import torch
from radonbridge.svd_basis import FixedChannelBasis,save_basis
from radonbridge.bridge import BridgeExchange,FeatureSpec
from radonbridge.projector import Projector
from radonbridge.model import PilotGraph
from radonbridge.optimization import configure_optimizer,clip_task_gradients
from check_integer_bridge import topology_check


def artifact(tmp,channels,key,seed=3416):
    # Full-rank anisotropic synthetic training features; no data files are read.
    generator=torch.Generator().manual_seed(seed)
    f=torch.randn(channels,channels*3,generator=generator,dtype=torch.float64)*torch.linspace(2.,.2,channels,dtype=torch.float64)[:,None]
    return save_basis(f@f.T,f.shape[1],tmp,key,seed,{'synthetic_test':True}),f


def basis_checks(tmp):
    ref,f=artifact(tmp,256,'cfp_stage3')
    before=torch.random.get_rng_state().clone()
    a=FixedChannelBasis(256,16,'cfp_stage3',ref)
    b=FixedChannelBasis(256,32,'cfp_stage3',ref)
    c=FixedChannelBasis(256,64,'cfp_stage3',ref)
    assert torch.equal(before,torch.random.get_rng_state())
    assert torch.equal(a.q,b.q[:,:16]) and torch.equal(b.q,c.q[:,:32])
    assert torch.allclose(c.q.T@c.q,torch.eye(64,dtype=torch.float64),atol=1e-12,rtol=0)
    u,s,_=torch.linalg.svd(f,full_matrices=False)
    assert torch.allclose(c.q@c.q.T,u[:,:64]@u[:,:64].T,atol=1e-11,rtol=1e-11)
    assert abs(c.metadata['retained_energy_ratio']-float(s[:64].square().sum()/s.square().sum()))<1e-12
    assert a.metadata['retained_energy_ratio']<b.metadata['retained_energy_ratio']<c.metadata['retained_energy_ratio']
    ref2,_=artifact(tmp,256,'cfp_stage3',3417)
    assert not torch.equal(b.q,FixedChannelBasis(256,32,'cfp_stage3',ref2).q)
    assert list(a.parameters())==[]
    exports=[x.float().export(Path(tmp)/'runtime') for x in [a,b,c]]
    assert len({x['artifact']['path'] for x in exports})==1
    assert len(list((Path(tmp)/'runtime').glob('*.npy')))==3
    for index,shape in enumerate([(4,5),(3,4,5)]):
        reference,_=artifact(tmp,6,'source'+str(index))
        basis=FixedChannelBasis(6,3,'source'+str(index),reference);p=Projector(shape,4,11)
        x=torch.randn(2,6,*shape,dtype=torch.float64)
        projected=p(x).reshape(2,6,4,11)
        assert torch.allclose(p(basis.encode(x)),basis.encode(projected).reshape(2,12,11),atol=1e-12,rtol=1e-12)
        y=torch.randn(2,12,11,dtype=torch.float64)
        assert torch.allclose(p.backproject(basis.decode(y.reshape(2,3,4,11)).reshape(2,24,11)),basis.decode(p.backproject(y)),atol=1e-12,rtol=1e-12)
    specs=[FeatureSpec('a',6,(3,4)),FeatureSpec('b',8,(3,4,3))]
    refs={spec.key:artifact(tmp,spec.channels,spec.key)[0] for spec in specs}
    kw=dict(M={'a':4,'b':8},S=11,rho={'a':.5,'b':.25},compression='fixed_svd_channel',basis_files=refs)
    m=BridgeExchange(specs,**kw).double();assert m.mixer.widths==(12,16)
    assert list(dict(m.named_parameters()))==['mixer.conv.weight']
    xs=[torch.randn(2,s.channels,*s.shape,dtype=torch.float64) for s in specs]
    assert torch.equal(m(*xs),torch.cat([x.flatten(1) for x in xs],1))
    torch.nn.init.normal_(m.mixer.conv.weight,std=.01)
    clone=BridgeExchange(specs,**kw).double();clone.load_state_dict(m.state_dict(),strict=True)
    assert torch.equal(m(*xs),clone(*xs))
    bad={k:v.clone() for k,v in m.state_dict().items()};bad['channel_bases.0.q'][0,0]+=1
    try:clone.load_state_dict(bad)
    except RuntimeError:pass
    else:raise AssertionError('Mismatched checkpoint basis accepted')
    for override in [{'basis_files':None},{'basis_files':{}},{'mode':'self'},{'compression':'typo'}]:
        try:BridgeExchange(specs,**(kw|override))
        except ValueError:pass
        else:raise AssertionError(override)
    return {'orthogonal':True,'nested_ranks':[16,32,64],'rng_unchanged':True,'energy_order_matches_svd':True,'shared_basis_across_ranks':True,'commutes_2d_3d':True,'only_mixer_trainable':True,'checkpoint_verified':True}


def checkpoint_optimizer_checks(tmp):
    c=torch.randn(1,2,3,224,224);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([1])
    g=PilotGraph(seed=3416);g.graph.eval()
    with torch.no_grad():expected={k:v.clone() for k,v in g.forward(c,o,y)[0].items()}
    native=g.save_state();del g
    cfg={'nodes':['cfp_stage3','oct_stage3'],'M':4,'S':11,'rho':.125,'mode':'radon','compression':'fixed_svd_channel','basis_files':{key:artifact(tmp,256,key,3418)[0] for key in ['cfp_stage3','oct_stage3']}}
    g=PilotGraph(seed=3416,bridge_configs=[cfg]);g.load_native_state(native);g.graph.eval()
    with torch.no_grad():actual=g.forward(c,o,y)[0]
    assert all(torch.equal(actual[k],expected[k]) for k in expected)
    opt=configure_optimizer(g,{'adapt_stages':[1,2,3,4],'training_regime':'full_finetune','backbone_lr':3e-5,'head_lr':1e-4,'bridge_lr':1e-4,'weight_decay':.01})
    bridge=g.modules_by_name()['bridge_0_exchange'];assigned=[p for group in opt.param_groups for p in group['params']]
    assert {id(p) for p in assigned}=={id(p) for p in g.graph.parameters()}
    assert all(id(basis.q) not in {id(p) for p in assigned} for basis in bridge.channel_bases)
    oldq=[b.q.clone() for b in bridge.channel_bases]
    tracked=[b for n,b in g.graph.named_buffers() if n.endswith('num_batches_tracked')];before=[b.clone() for b in tracked]
    first={k:next(m.parameters()).detach().clone() for k,m in g.modules_by_name().items() if list(m.parameters())}
    g.graph.train();g.forward(c,o,y);g.backward();clip_task_gradients(g,5);opt.step();opt.zero_grad(set_to_none=True)
    assert all(torch.equal(q,b.q) for q,b in zip(oldq,bridge.channel_bases))
    assert all(after>prior for after,prior in zip(tracked,before))
    assert all(not torch.equal(first[k],next(m.parameters())) for k,m in g.modules_by_name().items() if k in first)
    g.forward(c,o,y)
    octstem=next(g.modules_by_name()['oct_stage1'].parameters())
    cross=torch.autograd.grad(g.by_name['cfp_loss'].feature_message.current_state,octstem)[0]
    assert torch.isfinite(cross).all() and cross.abs().sum()>0
    return {'native_checkpoint_predictions_exact':True,'optimizer_exhaustive':True,'all_modules_update':True,'basis_unchanged':True,'batchnorm_updates':True,'cross_task_gradient':float(cross.norm())}

if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(777)
    with tempfile.TemporaryDirectory() as tmp:
        basis=basis_checks(tmp)
        refs={key:artifact(tmp,channels,key)[0] for key,channels in [('network0_stage1',2),('network1_stage2',3)]}
        topology=topology_check([2,3],compression='fixed_svd_channel',basis_files=refs)
        print(json.dumps({'basis':basis,'topology':topology,'checkpoint_optimizer':checkpoint_optimizer_checks(tmp),'passed':True}))
