"""Read-only full-MHD component switches; no features or participant IDs exported."""
import argparse,contextlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from .artifacts import relocate,resolve,sha256
from .data import PairedDataset
from .diagnostics import read_only,release_forward_graph
from .experiment import loader,write_json
from .metrics import classification_metrics
from .model import PilotGraph

STATES=((True,True),(True,False),(False,True),(False,False))
NAMES=('both_on','host_only','new_only','both_off')


@contextlib.contextmanager
def component_switch(g,host_on,new_on):
    modules=g.modules_by_name();host=modules['bridge_0_exchange'];new=modules['bridge_1_exchange']
    assert getattr(new,'delta_only',False) and 'bridge_parallel_merge' in modules
    handles=[]
    def identity(module,inputs,output):
        module.latest_deltas=tuple(torch.zeros_like(x) for x in inputs)
        return torch.cat([x.flatten(1) for x in inputs],1)
    def zero(module,inputs,output):
        module.latest_deltas=tuple(torch.zeros_like(x) for x in inputs)
        return torch.zeros_like(output)
    try:
        if not host_on:handles.append(host.register_forward_hook(identity))
        if not new_on:handles.append(new.register_forward_hook(zero))
        yield
    finally:
        for handle in handles:handle.remove()


def predict(g,data,limit=None):
    out={k:[] for k in ('cfp','oct')};ids=[];labels=[]
    with torch.no_grad():
        for index,(c,o,y,keys) in enumerate(loader(data,16,0)):
            if limit is not None and index>=limit:break
            logits,_=g.forward(c.cuda(),o.cuda(),y.cuda())
            for key in out:out[key].append(logits[key].softmax(1).cpu().numpy())
            ids.extend(keys);labels.extend(y.tolist());release_forward_graph(g)
    return dict(ids=np.asarray(ids),y=np.asarray(labels),**{k:np.concatenate(v) for k,v in out.items()})


def run(cfg,out,data_path):
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(9*1024**3/torch.cuda.get_device_properties(0).total_memory)
    cfg=relocate(cfg);checkpoint=Path(cfg['checkpoint']['path']);assert sha256(checkpoint)==cfg['checkpoint']['sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);c=saved['configuration']
    assert c['training_stage']=='host_augmentation' and not c.get('task_fusion') and len(c['bridges'])==2
    g=PilotGraph(bridge_configs=c['bridges'],seed=c['seed'],device='cuda');g.load_complete_state(saved['model'])
    original_ids={key:g.by_name[key].id for key in ('cfp_stage3','oct_stage3')}
    preflight=cfg.get('preflight',False);data=PairedDataset(data_path,'train' if preflight else 'validation',224)
    assert len(data)==(1264 if preflight else 296)
    start=time.monotonic();predictions={};records=[]
    with read_only(g):
        for name,(host_on,new_on) in zip(NAMES,STATES):
            with component_switch(g,host_on,new_on):predictions[name]=predict(g,data,1 if preflight else None)
            assert original_ids=={key:g.by_name[key].id for key in original_ids}
            v=predictions[name];np.savez(out/(name+'.npz'),**v)
            metrics={key:classification_metrics(v['y'],v[key]) for key in ('cfp','oct')}
            records.append(dict(state=name,host_on=host_on,new_on=new_on,metrics=metrics,
                                branch_mean_macro_f1=sum(metrics[k]['macro_f1'] for k in metrics)/2))
            write_json(out/'progress.json',dict(state=name,participants=len(v['y']),elapsed_seconds=time.monotonic()-start))
        replay=predict(g,data,1 if preflight else None)
        assert all(np.array_equal(predictions['both_on'][k],replay[k]) for k in replay),'Switch hooks were not restored'
        # Independently construct the updated native network; not the pretraining parent.
        state=g.save_state()
        del g
    # The read_only context restores its original graph before releasing it.
    import gc
    gc.collect();torch.cuda.empty_cache()
    native=PilotGraph(bridge_configs=[],seed=c['seed'],device='cuda')
    native.load_native_state({k:v for k,v in state.items() if k in native.definition.native_checkpoint_modules})
    with read_only(native):native_pred=predict(native,data,1 if preflight else None)
    assert all(np.allclose(predictions['both_off'][k],native_pred[k],rtol=1e-5,atol=1e-6) for k in ('cfp','oct'))
    assert np.array_equal(predictions['both_off']['ids'],native_pred['ids'])
    reference=cfg.get('selected_predictions')
    if not preflight:
        assert reference and sha256(reference['path'])==reference['sha256']
        with np.load(reference['path'],allow_pickle=False) as z:
            assert np.array_equal(z['ids'],replay['ids']) and np.array_equal(z['y'],replay['y'])
            assert all(np.allclose(z[k],replay[k],rtol=1e-5,atol=1e-6) for k in ('cfp','oct'))
    # Original independently pretrained probabilities are a distinct reference.
    parent_refs={}
    if not preflight:
        for branch,ref in c['parent_checkpoints'].items():
            p=Path(ref['path']).parent/'selected_predictions.npz'
            with np.load(p,allow_pickle=False) as z:
                assert np.array_equal(z['ids'],replay['ids']) and np.array_equal(z['y'],replay['y'])
                parent_refs[branch]=dict(prediction_sha256=sha256(p),metrics=classification_metrics(z['y'],z[branch]))
    result=dict(state='complete',passed=True,preflight=preflight,configuration=cfg,source_configuration=c,
                outcomes=records,original_parent_reference=parent_refs,parameters_buffers_gradients_rng_preserved=True,
                original_node_ids_preserved=True,both_off_matches_updated_native=True,both_on_matches_selected=not preflight,
                prediction_files={name:dict(path=str(out/(name+'.npz')),sha256=sha256(out/(name+'.npz'))) for name in NAMES},
                seconds=time.monotonic()-start,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,test_used=False,
                interpretation='Checkpoint functional dependence under inference interventions, not causal proof or retraining ablation')
    write_json(out/'summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--data',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    try:run(json.loads(Path(a.config).read_text()),out,a.data)
    except BaseException:
        import traceback
        write_json(out/'failure.json',dict(state='failed',error=traceback.format_exc()));raise
