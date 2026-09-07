"""Full-forward depth interventions and layer-specific train diagnostics."""
import argparse,contextlib,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from .artifacts import relocate,sha256
from .model import PilotGraph
from .data import PairedDataset
from .diagnostics import analyze_graph,read_only,release_forward_graph
from .component_ablation import predict
from .experiment import write_json
from .metrics import classification_metrics

@contextlib.contextmanager
def depth_switch(g,enabled):
    assert len(enabled)==len(g.communication_groups)
    handles=[]
    def identity(module,inputs,output):
        module.latest_deltas=tuple(torch.zeros_like(x) for x in inputs)
        return torch.cat([x.flatten(1) for x in inputs],1)
    try:
        for i,on in enumerate(enabled):
            if not on:handles.append(g.modules_by_name()[f'bridge_{i}_exchange'].register_forward_hook(identity))
        yield
    finally:
        for handle in handles:handle.remove()

def load_parents(g,c):
    for branch,ref in c['parent_checkpoints'].items():
        assert sha256(ref['path'])==ref['sha256']
        saved=torch.load(ref['path'],map_location='cpu',weights_only=False)
        g.load_native_state(saved['model'],branch)

def main(cfg,out,data_path):
    start=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise RuntimeError('Refuse to overwrite completed diagnostic')
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(9*1024**3/torch.cuda.get_device_properties(0).total_memory)
    cfg=relocate(cfg);ref=cfg['checkpoint'];assert sha256(ref['path'])==ref['sha256']
    saved=torch.load(ref['path'],map_location='cpu',weights_only=False);c=relocate(saved['configuration'])
    assert not c.get('task_fusion') and c['training_stage']=='communication'
    assert all(b['compression']=='fixed_svd_channel' and b['mode'] in ('radon','linear_resample') for b in c['bridges'])
    preflight=cfg.get('preflight',False);train=PairedDataset(data_path,'train',224)
    assert len(train)==1264
    g=PilotGraph(seed=c['seed'],bridge_configs=c['bridges'],device='cuda')
    load_parents(g,c);nodes={k:g.by_name[k].id for b in c['bridges'] for k in b['nodes']}
    results={}
    for phase in ('initial','selected'):
        if phase=='selected':g.load_complete_state(saved['model'])
        results[phase]={}
        for i,b in enumerate(c['bridges']):
            results[phase][str(i)]=analyze_graph(g,train,b['basis_files'],batch=16,
                probe_count=16 if preflight else 128,energy_limit=16 if preflight else 1264,
                exchange_name=f'bridge_{i}_exchange')
            release_forward_graph(g)
            write_json(out/'progress.json',dict(phase=phase,bridge=i,seconds=time.monotonic()-start))
    del saved
    data=train if preflight else PairedDataset(data_path,'validation',224)
    assert len(data)==(1264 if preflight else 296)
    n=len(c['bridges']);states=list(itertools.product((True,False),repeat=n)) if n>1 else [(True,)]
    predictions={};records=[]
    with read_only(g):
        for state in states:
            name=''.join('1' if x else '0' for x in state)
            with depth_switch(g,state):values=predict(g,data,1 if preflight else None)
            assert nodes=={k:g.by_name[k].id for k in nodes}
            predictions[name]=values
            np.savez(out/(name+'.npz'),**values)
            metrics={k:classification_metrics(values['y'],values[k]) for k in ('cfp','oct')}
            records.append(dict(state=name,enabled=state,metrics=metrics,
                branch_mean_macro_f1=sum(metrics[k]['macro_f1'] for k in metrics)/2,
                prediction_sha256=sha256(out/(name+'.npz'))))
        replay=predict(g,data,1 if preflight else None)
        assert all(np.array_equal(replay[k],predictions['1'*n][k]) for k in replay)
    if not preflight:
        ref=cfg['selected_predictions'];assert sha256(ref['path'])==ref['sha256']
        with np.load(ref['path'],allow_pickle=False) as z:
            assert np.array_equal(z['ids'],replay['ids']) and np.array_equal(z['y'],replay['y'])
            assert all(np.allclose(z[k],replay[k],rtol=1e-5,atol=1e-6) for k in ('cfp','oct'))
    if n>1:
        native=PilotGraph(seed=c['seed'],bridge_configs=[],device='cuda')
        state=g.save_state();native.load_native_state({k:v for k,v in state.items() if k in native.definition.native_checkpoint_modules})
        del g,state
        import gc
        gc.collect();torch.cuda.empty_cache()
        with read_only(native):p=predict(native,data,1 if preflight else None)
        assert all(np.allclose(p[k],predictions['0'*n][k],rtol=1e-5,atol=1e-6) for k in ('cfp','oct'))
    write_json(out/'summary.json',dict(state='complete',passed=True,preflight=preflight,diagnostics=results,
        interventions=records,full_forward_recomputed=True,original_node_ids_preserved=True,
        selected_prediction_replay_passed=not preflight,parameters_buffers_gradients_rng_preserved=True,
        multibridge_gradient_cosine_zero_not_assumed=True,seconds=time.monotonic()-start,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,test_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--data',required=True);a=p.parse_args()
    try:main(json.loads(Path(a.config).read_text()),a.output,a.data)
    except Exception as exc:
        import traceback
        write_json(Path(a.output)/'failure.json',dict(state='oom' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'failed',error=traceback.format_exc()))
        raise
