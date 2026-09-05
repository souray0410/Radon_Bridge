"""Read-only, train-only checkpoint diagnostics; never updates an optimizer."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import Subset
from .experiment import loader,write_json
from .model import PilotGraph
from .data import PairedDataset
from .svd_basis import FixedChannelBasis,file_sha


def state_hash(g):
    h=hashlib.sha256()
    for name,m in sorted(g.modules_by_name().items()):
        for key,t in sorted(m.state_dict().items()):h.update((name+'/'+key).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def read_only(g):
    before=state_hash(g);flags={m:m.training for m in g.graph.modules()}
    py=random.getstate();npstate=np.random.get_state();cpu=torch.random.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    grads={id(p):None if p.grad is None else p.grad.detach().cpu().clone() for p in g.graph.parameters()}
    scale=g.modules_by_name()['task_loss_sum'].scale
    try:
        g.graph.eval();yield
    finally:
        unchanged=state_hash(g)==before;grads_unchanged=True
        for p in g.graph.parameters():
            old=grads[id(p)]
            grads_unchanged &= (p.grad is None) if old is None else (p.grad is not None and torch.equal(p.grad.cpu(),old))
        for m,flag in flags.items():m.training=flag
        g.modules_by_name()['task_loss_sum'].scale=scale
        random.setstate(py);np.random.set_state(npstate);torch.random.set_rng_state(cpu)
        if cuda:torch.cuda.set_rng_state_all(cuda)
        assert unchanged and grads_unchanged,'Diagnostic changed parameters, buffers or parameter gradients'


def cosine(a,b):
    an=float(a.norm());bn=float(b.norm())
    return {'cfp_norm':an,'oct_norm':bn,'cosine':float(torch.dot(a,b)/(an*bn)) if an>0 and bn>0 else None,
            'undefined_reason':None if an>0 and bn>0 else 'zero_gradient_norm'}


def release_forward_graph(g):
    """Release diagnostic activations after both CE gradients were collected.

    autograd.grad only traverses requested stage3/bridge parameters. Saved
    activations upstream of them can therefore survive, including the CFP
    graph retained for the first CE. MHD keeps node values and a forward trace;
    detach those caches between probe batches, without changing model state.
    """
    g.graph._forward_trace.clear()
    for node in g.nodes:
        node.feature_message.current_state=node.feature_message.current_state.detach()
    for module in g.modules_by_name().values():
        for attr in ['latest_inputs','latest_deltas']:
            if hasattr(module,attr):
                setattr(module,attr,[v.detach() for v in getattr(module,attr)])


def learned_rowspace(codec,device):
    # Separate subspace retention from learned encoder amplification.
    from types import SimpleNamespace
    from .svd_basis import tensor_sha
    e=codec.encoder.weight.detach().squeeze(-1).cpu().double()
    d=codec.decoder.weight.detach().squeeze(-1).cpu().double()
    _,singular,vh=torch.linalg.svd(e,full_matrices=False)
    tolerance=max(e.shape)*torch.finfo(e.dtype).eps*float(singular.max())
    rank=int((singular>tolerance).sum());q=vh[:rank].T.contiguous().to(device)
    metadata={'role':'orthogonal projector onto current learned encoder row space; not encoder amplitude',
        'numerical_rank':rank,'rank_tolerance':tolerance,'encoder_singular_values':singular.tolist(),
        'encoder_sha256':tensor_sha(e),'decoder_sha256':tensor_sha(d),
        'decoder_minus_encoder_transpose_l2':float((d-e.T).norm()),'initialization':codec.metadata}
    return SimpleNamespace(q=q,encode=lambda x:torch.einsum('cr,bc...->br...',q.to(x),x),metadata=metadata)


def analyze_graph(g,data,basis_refs,batch=16,probe_count=128,energy_limit=None):
    device=next(g.graph.parameters()).device;modules=g.modules_by_name();exchange=modules.get('bridge_0_exchange')
    keys=['cfp_stage3','oct_stage3'];bases={}
    for key in keys:
        if exchange is not None and exchange.compression=='learned_channel':
            bases[key]=learned_rowspace(exchange.channel_codecs[exchange.keys.index(key)],device)
        elif exchange is not None and exchange.compression!='learned_projected':
            bases[key]=exchange.channel_bases[exchange.keys.index(key)]
        else:bases[key]=FixedChannelBasis(256,32,key,basis_refs[key]).to(device=device,dtype=torch.float32)
    accum={k:{'input_energy':0.,'retained_energy':0.,'delta_energy':0.} for k in keys}
    sums={k:torch.zeros(bases[k].q.shape[0],dtype=torch.float64) for k in keys};projected_sums={k:torch.zeros(bases[k].q.shape[1],dtype=torch.float64) for k in keys};counts={k:0 for k in keys}
    groups={k:list(modules[k].parameters()) for k in keys}
    if exchange is not None:
        for name,m in exchange.named_modules():
            ps=list(m.parameters(recurse=False))
            if ps:groups['bridge_0_exchange.'+name]=ps
    flat=[];slices={};start=0
    for key,ps in groups.items():
        flat.extend(ps);count=sum(p.numel() for p in ps);slices[key]=(start,start+count);start+=count
    assert len({id(p) for p in flat})==len(flat)
    gradients={branch:torch.zeros(start,dtype=torch.float64) for branch in ['cfp','oct']}
    indices=sorted(range(len(data)),key=lambda i:data.rows[i]['order'])[:probe_count]
    assert len(indices)==probe_count
    probe_ids=[data.rows[i]['order'] for i in indices]
    with read_only(g):
        with torch.no_grad():
            energy_data=data if energy_limit is None else Subset(data,list(range(energy_limit)))
            for c,o,y,_ in loader(energy_data,batch,0):
                g.forward(c.to(device),o.to(device),y.to(device))
                for i,key in enumerate(keys):
                    x=exchange.latest_inputs[exchange.keys.index(key)] if exchange is not None else g.by_name[key].feature_message.current_state
                    dx=exchange.latest_deltas[exchange.keys.index(key)] if exchange is not None else None
                    values=accum[key];values['input_energy']+=float(x.double().square().sum());values['retained_energy']+=float(bases[key].encode(x).double().square().sum())
                    if exchange is not None and exchange.compression=='learned_channel':
                        codec=exchange.channel_codecs[exchange.keys.index(key)]
                        values['encoded_energy']=values.get('encoded_energy',0.)+float(codec.encode(x).double().square().sum())
                    if dx is not None:values['delta_energy']+=float(dx.double().square().sum())
                    f=x.detach().movedim(1,0).reshape(x.shape[1],-1).double();q=bases[key].q.double()
                    v=f.sum(1);sums[key]+=v.cpu();projected_sums[key]+=(q.T@v).cpu();counts[key]+=f.shape[1]
                    del f,q,v
        for c,o,y,_ in loader(Subset(data,indices),batch,0):
            g.forward(c.to(device),o.to(device),y.to(device))
            for i,branch in enumerate(['cfp','oct']):
                loss=g.by_name[branch+'_loss'].feature_message.current_state
                gs=torch.autograd.grad(loss,flat,retain_graph=i==0,allow_unused=True)
                offset=0
                for p,grad in zip(flat,gs):
                    if grad is not None:
                        assert torch.isfinite(grad).all()
                        gradients[branch][offset:offset+p.numel()].add_(grad.detach().cpu().flatten().double(),alpha=len(y)/probe_count)
                    offset+=p.numel()
                del gs,grad,loss
            release_forward_graph(g)
    result={key:cosine(gradients['cfp'][lo:hi],gradients['oct'][lo:hi]) for key,(lo,hi) in slices.items()}
    if exchange is not None and exchange.compression!='learned_projected':
        v=result['bridge_0_exchange.mixer.conv'];assert v['cosine'] is None or abs(v['cosine'])<1e-10,'Fixed single-bridge output-row support is not disjoint'
    for key,values in accum.items():
        den=values['input_energy'];values['retained_energy_ratio']=values['retained_energy']/den if den>0 else None
        values['delta_over_input_l2']=(values['delta_energy']/den)**.5 if den>0 else None
        mean_energy=float(sums[key].square().sum())/counts[key];retained_mean_energy=float(projected_sums[key].square().sum())/counts[key]
        variance_energy=max(0.,den-mean_energy);retained_variance=max(0.,values['retained_energy']-retained_mean_energy)
        values.update(mean_energy_fraction=mean_energy/den if den>0 else None,retained_variance_ratio=retained_variance/variance_energy if variance_energy>0 else None,centering_scope='global training channel mean; not per participant')
        if 'encoded_energy' in values:values['encoded_energy_ratio']=values['encoded_energy']/den if den>0 else None
        values['projection_basis']=bases[key].metadata
        values['projection_role']='actual channel compression' if exchange is not None and exchange.compression!='learned_projected' else 'parent SVD diagnostic subspace; not the learned CM map'
        if exchange is not None and exchange.compression=='learned_channel':
            values['projection_role']='current encoder row-space orthogonal projection; encoded_energy_ratio separately measures scaling'
    return {'energy':accum,'gradient_groups':result,'probe_ids':probe_ids,'probe_ids_sha256':hashlib.sha256(json.dumps(probe_ids,separators=(',',':')).encode()).hexdigest(),
            'probe_participants':probe_count,'energy_participants':len(data) if energy_limit is None else energy_limit,
            'state_parameters_bn_gradients_rng_preserved':True,'test_used':False}


def main(cfg,out,data_path):
    start=time.monotonic();torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(9*1024**3/torch.cuda.get_device_properties(0).total_memory)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise RuntimeError('Diagnostic already exists')
    trial=Path(cfg['trial_directory']);config=json.loads((trial/'configuration.json').read_text());train=PairedDataset(data_path,'train',224);assert len(train)==1264
    g=PilotGraph(seed=config['seed'],bridge_configs=config['bridges'],device='cuda')
    for branch,parent in config['parent_checkpoints'].items():
        assert file_sha(parent['path'])==parent['sha256'];saved=torch.load(parent['path'],map_location='cpu',weights_only=False);g.load_native_state(saved['model'],branch)
    results={};preflight=cfg.get('preflight',False)
    for phase in ['initial','selected']:
        if phase=='selected':
            checkpoint=trial/cfg.get('selected_file','selected.pt');expected=cfg['selected_sha256'];assert file_sha(checkpoint)==expected
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert set(saved['model'])==set(g.modules_by_name())
            for key,module in g.modules_by_name().items():module.load_state_dict(saved['model'][key],strict=True)
            del saved
        results[phase]=analyze_graph(g,train,cfg['basis_files'],batch=16,probe_count=cfg.get('probe_count',16 if preflight else 128),energy_limit=cfg.get('energy_limit',16 if preflight else None))
        write_json(out/'progress.json',{'completed_phase':phase,'seconds':time.monotonic()-start})
    write_json(out/'summary.json',{'state':'complete','passed':True,'preflight':preflight,'trial_directory':str(trial),'selected_sha256':cfg['selected_sha256'],
        'phases':results,'seconds':time.monotonic()-start,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'test_used':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--data',required=True);a=p.parse_args()
    try:main(json.loads(Path(a.config).read_text()),a.output,a.data)
    except BaseException:
        import traceback
        write_json(Path(a.output)/'failure.json',{'state':'failed','error':traceback.format_exc()});raise
