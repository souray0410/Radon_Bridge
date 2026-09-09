"""Read-only latent Radon atlas; individual examples stay in restricted storage."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from radon_bridge.runtime.artifacts import resolve, sha256
from radon_bridge.data.dataset import PairedDataset
from radon_bridge.evaluation.replay import state_hash
from radon_bridge.analysis.diagnostics import read_only, release_forward_graph
from radon_bridge.training.trainer import loader, write_json
from radon_bridge.evaluation.evaluator import load_model, arrays, validate_prediction

PACKETS=('projection','self_message','cross_message','total_message')
SPATIAL=('input','self_return','cross_return','total_return','updated')


def angle_display(matrix,directions):
    """Return angle-sorted 2D rows with the required s reversal under n -> -n."""
    directions=np.asarray(directions);theta=np.mod(np.arctan2(directions[:,1],directions[:,0]),np.pi)
    canonical=np.stack([np.cos(theta),np.sin(theta)],axis=1)
    flip=(canonical*directions).sum(1)<0
    values=np.array(matrix,copy=True);values[flip]=values[flip,::-1]
    order=np.argsort(theta)
    return values[order],theta[order],dict(order=order.tolist(),s_reversed=flip.tolist())


def energy_terms(x,a,b):
    n=x.shape[0]//2
    flatten=lambda v:v.reshape(n,-1).double()
    x,a,b=map(flatten,(x,a,b));ex=x.square().sum(1);ea=a.square().sum(1);eb=b.square().sum(1)
    dot=(a*b).sum(1);et=(a+b).square().sum(1)
    valid=ex>0;cross_valid=(ea>0)&(eb>0)
    ratio=lambda e:torch.where(valid,(e/ex.clamp_min(1e-300)).sqrt(),torch.nan)
    return dict(self_relative=ratio(ea),cross_relative=ratio(eb),total_relative=ratio(et),
        self_cross_cosine=torch.where(cross_valid,dot/(ea*eb).clamp_min(1e-300).sqrt(),torch.nan),
        energy_identity_relative_error=(et-ea-eb-2*dot).abs()/torch.maximum(et,ea+eb).clamp_min(1e-300))


def block_decompose(mixer,encoded):
    if len(encoded)!=2 or getattr(mixer,'s_axis_permutation',None) is not None:
        raise ValueError('Atlas is restricted to two sources without S-axis permutation')
    offsets=np.cumsum([0]+list(mixer.widths));w=mixer.conv.weight*mixer.mask
    parts=[]
    for dst in (0,1):
        blocks=[]
        for src in (dst,1-dst):
            blocks.append(F.conv1d(encoded[src],w[offsets[dst]:offsets[dst+1],offsets[src]:offsets[src+1]],padding=mixer.conv.kernel_size[0]//2))
        parts.append(tuple(blocks))
    return parts


def check_linear_sum(a,b,total):
    error=float((a+b-total).detach().abs().max());scale=float(total.detach().abs().max())
    if error>2e-5+2e-4*scale:raise ValueError('Side-path block reconstruction differs from actual execution')
    return dict(max_absolute=error,reference_max_absolute=scale)


class Aggregator:
    def __init__(self):self.records={};self.errors=[]
    def update(self,key,meta,packet,spatial,scalar,labels):
        # Average the two eyes and channels within each participant, then participants equally.
        n=len(labels);values={}
        for name,t in packet.items():
            t=t.reshape(n,2,-1,meta['M'],meta['S']).double()
            values[name+'_ms']=t.square().mean((1,2)).cpu().numpy()
            power=torch.fft.rfft(t,dim=-1,norm='ortho').abs().square()
            power[...,1:-1 if meta['S']%2==0 else None]*=2
            values[name+'_spectrum']=power.mean((1,2,3)).cpu().numpy()
        for name,t in spatial.items():
            values[name+'_ms']=t.reshape(n,2,-1,*meta['shape']).double().square().mean((1,2)).cpu().numpy()
        scalars={k:v.cpu().numpy() for k,v in scalar.items()}
        for group,mask in [('all',np.ones(n,dtype=bool)),('label0',labels==0),('label1',labels==1)]:
            count=int(mask.sum())
            if not count:continue
            r=self.records.setdefault(key+'/'+group,dict(metadata=meta,group=group,count=0,sums={},scalars={}))
            r['count']+=count
            for name,arr in values.items():r['sums'][name]=r['sums'].get(name,0)+arr[mask].sum(0)
            for name,arr in scalars.items():
                v=arr[mask];v=v[np.isfinite(v)]
                st=r['scalars'].setdefault(name,dict(count=0,sum=0.,sum2=0.))
                st['count']+=len(v);st['sum']+=float(v.sum());st['sum2']+=float((v*v).sum())
    def export(self):
        result={}
        for key,r in self.records.items():
            stats={}
            for name,s in r['scalars'].items():
                n=s['count'];mean=s['sum']/n if n else None
                stats[name]=dict(mean=mean,sample_sd=float(np.sqrt(max(0.,(s['sum2']-s['sum']**2/n)/(n-1)))) if n>1 else None,
                    defined_participants=n,undefined_participants=r['count']-n)
            result[key]=dict(metadata=r['metadata'],group=r['group'],participants=r['count'],
                            arrays={name:(v/r['count']).tolist() for name,v in r['sums'].items()},scalars=stats)
        return result


def sample_ids(data,count=16):return sorted(r['order'] for r in data.rows)[:count]


def attach_collectors(g,agg,context,private):
    handles=[];captured={}
    for i,group in enumerate(g.communication_groups):
        ex=g.modules_by_name()[group['exchange_edge_name']]
        if ex.compression not in ('fixed_svd_channel','fixed_random_orthogonal_channel') or ex.mode not in ('radon','linear_resample'):
            raise ValueError('Unexpected atlas bridge')
        def mixer_hook(module,inputs,output,index=i):captured[index]=(tuple(v.detach() for v in inputs),tuple(v.detach() for v in output))
        handles.append(ex.mixer.register_forward_hook(mixer_hook))
        def collect(module,inputs,output,index=i):
            encoded,mixed=captured.pop(index);parts=block_decompose(module.mixer,encoded)
            labels=context['labels'];ids=context['ids'];n=len(ids)
            for dst,(x,(own,cross),actual) in enumerate(zip(inputs,parts,mixed)):
                q=module.channel_bases[dst];projector=module.projectors[dst]
                self_return=q.decode(projector.backproject(own));cross_return=q.decode(projector.backproject(cross))
                delta=module.latest_deltas[dst]
                agg.errors.append(dict(packet=check_linear_sum(own,cross,actual),returned=check_linear_sum(self_return,cross_return,delta)))
                meta=dict(source=module.keys[dst],bridge=index,shape=list(x.shape[2:]),C=x.shape[1],r=q.q.shape[1],
                    M=projector.M,S=projector.S,k=module.mixer.conv.kernel_size[0],mode=module.mode,
                    compression=module.compression,geometry=projector.metadata)
                packets=dict(zip(PACKETS,(encoded[dst],own,cross,actual)))
                spatial=dict(zip(SPATIAL,(x,self_return,cross_return,delta,x+delta)))
                scalar=energy_terms(x,self_return,cross_return)
                c=q.encode(x);ex2=x.reshape(n,-1).double().square().sum(1);ec=c.reshape(n,-1).double().square().sum(1)
                scalar['channel_energy_retained']=torch.where(ex2>0,ec/ex2,torch.nan)
                key=f'bridge{index}_{module.keys[dst]}'
                agg.update(key,meta,packets,spatial,scalar,labels)
                # Every selected person keeps both eyes and the same predeclared channel prefix.
                for pos,identifier in enumerate(ids):
                    if identifier not in context['sample_ids']:continue
                    prefix=min(3,q.q.shape[1]);record={}
                    for name,t in packets.items():record[name]=t.reshape(n,2,-1,projector.M,projector.S)[pos,:,:prefix].cpu().numpy()
                    record['compressed_native']=c.reshape(n,2,-1,*x.shape[2:])[pos,:,:prefix].cpu().numpy()
                    record.update({name+'_rms':t.reshape(n,2,-1,*x.shape[2:])[pos].square().mean(1).sqrt().cpu().numpy() for name,t in spatial.items()})
                    np.savez_compressed(private/(identifier+'__'+key+'.npz'),**record)
        handles.append(ex.register_forward_hook(collect))
    return handles


def initialize_reference(g,record):
    for branch,ref in record['configuration']['parent_checkpoints'].items():
        if sha256(ref['path'])!=ref['sha256']:raise ValueError('Parent checkpoint changed')
        state=torch.load(resolve(ref['path']),map_location='cpu',weights_only=False)
        g.load_native_state(state['model'],branch=branch)
    with torch.no_grad():
        for group in g.communication_groups:g.modules_by_name()[group['exchange_edge_name']].mixer.conv.weight.zero_()


def process(job,lock,out,preflight):
    tick=time.monotonic();manifest=json.loads((lock/'candidate_lock.json').read_text())
    for name,h in manifest['files'].items():
        if sha256(lock/name)!=h:raise ValueError('Atlas registration changed')
    if job not in json.loads((lock/'jobs.json').read_text()):raise ValueError('Unregistered atlas job')
    for phase,paths in job['expected_predictions'].items():
        for split,path in paths.items():
            if sha256(path)!=job['expected_prediction_sha256'][phase][split]:raise ValueError('Reference prediction changed')
    parent=Path(manifest['after_run']);status=json.loads((parent/'status.json').read_text())
    if status['state']!='complete':raise ValueError('Predecessor test and statistics have not completed')
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(9*1024**3/torch.cuda.get_device_properties(0).total_memory)
    all_aggregates={};prediction_files={};evidence=[]
    for phase in ('constructed_initial','selected'):
        g=load_model(job['model'])
        if phase=='constructed_initial':initialize_reference(g,job['model'])
        g.graph.eval();before=state_hash(g);node_ids={n.name:n.id for n in g.nodes}
        for split in (('train',) if preflight else ('train','validation','test')):
            root=Path(manifest['data_directories']['test' if split=='test' else 'train_development'])
            if sha256(root/'selected.csv')!=manifest['data_selected_sha256']['test' if split=='test' else 'train_development']:
                raise ValueError('Cohort manifest changed')
            data=PairedDataset(str(root),split,224)
            expected={'train':1264,'validation':296,'test':290}[split]
            if len(data)!=expected:raise ValueError('Wrong split size')
            ids_selected=sample_ids(data);private=out/'private'/phase/split;private.mkdir(parents=True,mode=0o700)
            write_json(private/'sample_manifest.json',dict(ids=ids_selected,selection='first16 sorted stable cache order, label/prediction independent',channels=[0,1,2],eyes=['left','right']))
            agg=Aggregator();context=dict(sample_ids=set(ids_selected));handles=attach_collectors(g,agg,context,private)
            probs={'cfp':[],'oct':[]};ids=[];ys=[]
            try:
                with read_only(g),torch.no_grad():
                    for batch,(c,o,y,keys) in enumerate(loader(data,16,0)):
                        if preflight and batch>=1:break
                        if preflight and batch==0:probe=(c,o,y)
                        context.update(labels=y.numpy(),ids=keys)
                        logits,_=g.forward(c.cuda(),o.cuda(),y.cuda())
                        for b in probs:probs[b].append(logits[b].softmax(1).cpu().numpy())
                        ids.extend(keys);ys.extend(y.tolist());release_forward_graph(g)
                        if batch%8==0:write_json(out/'progress.json',dict(phase=phase,split=split,participants=len(ids),preflight=preflight,seconds=time.monotonic()-tick))
            finally:
                for h in handles:h.remove()
            if state_hash(g)!=before or node_ids!={n.name:n.id for n in g.nodes}:raise ValueError('Read-only state or Node ID changed')
            value=dict(ids=np.asarray(ids),y=np.asarray(ys),**{b:np.concatenate(v) for b,v in probs.items()})
            if preflight:
                with read_only(g),torch.no_grad():
                    replay,_=g.forward(*(v.cuda() for v in probe))
                    for b in probs:
                        if not np.array_equal(replay[b].softmax(1).cpu().numpy(),value[b]):raise ValueError('Hooks changed train preflight output')
                    release_forward_graph(g)
                    if phase=='constructed_initial':
                        native,_=g.native_forward(*(v.cuda() for v in probe))
                        for b in probs:
                            if not np.allclose(native[b].softmax(1).cpu().numpy(),value[b],atol=1e-6,rtol=1e-5):raise ValueError('Initial bridge differs from native prediction')
                        release_forward_graph(g)
                del probe
            if not preflight:validate_prediction(value,data)
            if not preflight and split in ('validation','test'):
                ref=arrays(job['expected_predictions'][phase][split])
                if any(not np.array_equal(value[k],ref[k]) for k in ('ids','y')):raise ValueError('Prediction identity mismatch')
                for b in probs:
                    if not np.allclose(value[b],ref[b],rtol=1e-5,atol=1e-6) or not np.array_equal(value[b].argmax(1),ref[b].argmax(1)):
                        raise ValueError('Atlas hooks changed selected/parent predictions')
            file=private/'predictions.npz';np.savez(file,**value);prediction_files[str(file.relative_to(out))]=sha256(file)
            private_hashes={str(p.relative_to(out)):sha256(p) for p in sorted(private.glob('*')) if p.is_file()}
            write_json(private/'artifact_manifest.json',private_hashes)
            prediction_files[str((private/'artifact_manifest.json').relative_to(out))]=sha256(private/'artifact_manifest.json')
            exported=agg.export();all_aggregates[phase+'/'+split]=exported
            if not all(v['participants']==(len(ids) if v['group']=='all' else sum(value['y']==int(v['group'][-1]))) for v in exported.values()):
                raise ValueError('Participant weighting error')
            evidence.append(dict(phase=phase,split=split,participants=len(ids),sample_manifest_sha256=sha256(private/'sample_manifest.json'),
                linear_recomposition_checks=len(agg.errors),max_packet_error=max(e['packet']['max_absolute'] for e in agg.errors),
                max_return_error=max(e['returned']['max_absolute'] for e in agg.errors),reference_probability_check=(not preflight and split!='train')))
        del g
        import gc
        gc.collect();torch.cuda.empty_cache()
    write_json(out/'aggregates.json',all_aggregates);prediction_files['aggregates.json']=sha256(out/'aggregates.json')
    write_json(out/'summary.json',dict(state='accepted',job_id=job['job_id'],kind='sinogram_atlas',split='preflight' if preflight else 'all',
        source_commit=os.environ.get('RB_EVALUATION_COMMIT'),strict_model_load=True,parameters_BN_gradients_RNG_preserved=True,
        original_node_ids_preserved=True,test_used=not preflight,post_hoc_descriptive=True,training=False,
        candidate_lock_sha256=sha256(lock/'candidate_lock.json'),prediction_files=prediction_files,evidence=evidence,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,seconds=time.monotonic()-tick,updated_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('record','lock','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--split',choices=('preflight','all'),required=True);a=p.parse_args()
    try:
        if (a.output/'summary.json').exists():raise ValueError('Do not overwrite accepted atlas')
        process(json.loads(a.record.read_text()),a.lock,a.output,a.split=='preflight')
    except BaseException:
        write_json(a.output/'failure.json',dict(state='needs_attention',error=traceback.format_exc(),updated_at=time.time()));raise
