"""Bounded factorial, balanced estimands, multilevel MHD and intervention invariants."""
import itertools,json,tempfile,sys
from pathlib import Path
import torch
from radonbridge.depth_study import matrices,definitions,apply_clipping,arithmetic,SEEDS,LRS
from radonbridge.geometry_study import configuration
from radonbridge.pretest_study import row
from radonbridge.svd_basis import save_basis
from radonbridge.model import PilotGraph
from radonbridge.optimization import configure_optimizer,clip_task_gradients
from radonbridge.diagnostics import release_forward_graph,read_only,state_hash,analyze_graph
from radonbridge.depth_diagnostics import depth_switch

def matrix_test():
    from radonbridge.geometry_study import cost
    for mode in ('radon','linear_resample'):
        assert arithmetic((3,),16,mode)['flops_per_participant']==cost(32,64,3,16,mode)['flops_per_participant']
    parents={s:{k:dict(path='/parent/'+k,sha256=str(s)+k) for k in ('cfp','oct')} for s in SEEDS}
    bases={s:{f'{b}_stage{stage}':dict(path='/basis/'+b+str(stage),sha256=str(s)+b+str(stage)) for b in ('cfp','oct') for stage in (2,3,4)} for s in SEEDS}
    rows=matrices(parents,bases);assert len(rows)==108
    baseline=[row(f'none_{s}_{lr}',configuration(dict(family='none'),s,lr,'branch',parents[s],{}),dict(id='no_communication'),'baseline') for s in SEEDS for lr in LRS]
    defs=definitions(rows,baseline);assert len(defs)==60
    assert sum(d['primary'] for d in defs)==22
    lookup={r['id']:r for r in rows+baseline}
    for d in defs:
        assert abs(sum(d['weights'].values()))<1e-12
        for seed in SEEDS:
            for lr in LRS:assert abs(sum(v for k,v in d['weights'].items() if lookup[k]['seed']==seed and lookup[k]['backbone_lr']==lr))<1e-12
    reused=[r for r in rows if r['structure']['stages']==[3] and r['structure']['r']==16]
    assert len(reused)==12 and len(rows)-len(reused)==96
    for n,r in [(2,23),(3,28)]:assert abs(r*r/(n*16**2)-1)<.05
    # Additive layer effects have zero pairwise/three-way interaction.
    scores={r['id']:sum(r['structure'].get('stages',[])) for r in rows+baseline}
    for d in defs:
        if 'interaction' in d['id']:assert abs(sum(v*scores[k] for k,v in d['weights'].items()))<1e-10

def topology_test():
    torch.set_num_threads(3)
    with tempfile.TemporaryDirectory() as t:
        refs={}
        for stage in (2,3,4):
            c=64*2**(stage-1)
            for b in ('cfp','oct'):
                key=f'{b}_stage{stage}';refs[key]=save_basis(torch.eye(c,dtype=torch.float64),1,t,key,3416,{'synthetic':True})
        for mode in ('radon','linear_resample'):
            bridges=[dict(nodes=[f'{b}_stage{s}' for b in ('cfp','oct')],M=32,S=64,kernel_size=3,r=16,h=512,rho=16/(64*2**(s-1)),mode=mode,
                compression='fixed_svd_channel',basis_files={f'{b}_stage{s}':refs[f'{b}_stage{s}'] for b in ('cfp','oct')}) for s in (2,3,4)]
            g=PilotGraph(seed=3416,bridge_configs=bridges);g.graph.eval()
            c=torch.randn(1,2,3,224,224);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([1])
            nodes={key:g.by_name[key].id for key in refs}
            with torch.no_grad():
                a,_=g.forward(c,o,y);a={k:v.clone() for k,v in a.items()}
                for s in (2,3,4):
                    shape=arithmetic((s,),16,mode)['bridges'][0]
                    for branch in ('cfp','oct'):
                        assert list(g.by_name[f'{branch}_stage{s}'].feature_message.current_state.shape[2:])==shape[branch+'_shape']
                with depth_switch(g,(False,False,False)):b,_=g.forward(c,o,y)
                assert all(torch.equal(a[k],b[k]) for k in a)
            cfg=dict(bridges=bridges,communication_clipping='per_bridge');apply_clipping(g,cfg)
            opt=configure_optimizer(g,dict(adapt_stages=[1,2,3,4],backbone_lr=6e-5,head_lr=1e-4,bridge_lr=1e-4,training_regime='full_finetune'))
            for _ in range(2):
                opt.zero_grad(set_to_none=True);g.graph.train();g.forward(c,o,y);g.backward()
                norms=clip_task_gradients(g,5.);assert set(norms)=={'cfp','oct','bridge_0','bridge_1','bridge_2'}
                assert all(g.modules_by_name()[f'bridge_{i}_exchange'].mixer.conv.weight.grad.norm()>0 for i in range(3))
                opt.step();release_forward_graph(g)
            opt.zero_grad(set_to_none=True);g.graph.eval();g.forward(c,o,y);g.backward()
            grads={n:[None if p.grad is None else p.grad.clone() for p in m.parameters()] for n,m in g.modules_by_name().items()}
            opt.zero_grad(set_to_none=True);_,loss=g.native_forward(c,o,y);loss.backward()
            for n,m in g.modules_by_name().items():
                for old,p in zip(grads[n],m.parameters()):
                    assert (old is None)==(p.grad is None)
                    if old is not None:assert torch.allclose(old,p.grad,atol=3e-5,rtol=3e-4),n
            opt.zero_grad(set_to_none=True);release_forward_graph(g)
            state=g.save_state();before=state_hash(g);g.load_complete_state(state)
            with read_only(g),torch.no_grad():
                for enabled in itertools.product((True,False),repeat=3):
                    with depth_switch(g,enabled):out,_=g.forward(c,o,y)
                    assert nodes=={key:g.by_name[key].id for key in nodes}
                    for i,on in enumerate(enabled):
                        if not on:assert all(torch.count_nonzero(x)==0 for x in g.modules_by_name()[f'bridge_{i}_exchange'].latest_deltas)
                    release_forward_graph(g)
                out,_=g.forward(c,o,y);saved={k:v.clone() for k,v in out.items()};g.load_complete_state(state);other,_=g.forward(c,o,y)
                assert all(torch.equal(saved[k],other[k]) for k in saved)
            assert state_hash(g)==before
            class Data(torch.utils.data.Dataset):
                rows=[{'order':0},{'order':1}]
                def __len__(self):return 2
                def __getitem__(self,i):return c[0],o[0],torch.tensor(i),i
            for i,bridge in enumerate(bridges):
                d=analyze_graph(g,Data(),bridge['basis_files'],batch=1,probe_count=2,energy_limit=2,exchange_name=f'bridge_{i}_exchange')
                assert set(d['energy'])==set(bridge['nodes'])
                assert d['state_parameters_bn_gradients_rng_preserved']
                release_forward_graph(g)
            del g,opt,grads,state
    return True

if __name__=='__main__':
    matrix_test();report=dict(passed=True,new_training=96,reused_stage3=12,reused_baseline=6,primary=22,secondary=38,test_used=False)
    if '--matrix-only' not in sys.argv:report['triple_mhd_verified']=topology_test()
    print(json.dumps(report))
