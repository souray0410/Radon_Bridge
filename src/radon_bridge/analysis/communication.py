"""Read-only paired-message perturbations and controlled inference timing.

Participant-indexed files stay in the authorized run directory. The public
report contains only aggregate statistics and hashes, never cached features.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from radon_bridge.data.dataset import PairedDataset
from radon_bridge.analysis.diagnostics import read_only, release_forward_graph
from radon_bridge.training.trainer import loader, write_json
from radon_bridge.evaluation.metrics import classification_metrics
from radon_bridge.models.model import PilotGraph
from radon_bridge.methods.basis import file_sha


def derangements(n,count=20,seed=202609051):
    rng=np.random.default_rng(seed);perms=[]
    while len(perms)<count:
        p=rng.permutation(n)
        if np.all(p!=np.arange(n)):perms.append(p)
    return np.stack(perms)


def cached_forward(g,features,y):
    ex=g.modules_by_name()['bridge_0_exchange'];edge=next(e for e in g.edges if e.name=='bridge_0_exchange')
    level=g.builder.forward_edge_levels[edge.id]
    g.builder.set_inputs(dict(zip(ex.keys,features))|{'target':y})
    g.graph.forward(levels=[l for l in g.forward_levels if l>=level])
    return {b:g.by_name[b+'_logits'].feature_message.current_state.softmax(1) for b in g.branches}


def block_messages(mixer,features,overrides):
    """Overrides (destination,source) only; None deletes that cross term."""
    offsets=np.cumsum([0]+list(mixer.widths));weight=mixer.conv.weight*mixer.mask
    outputs=[]
    for dst in range(len(features)):
        total=None
        for src in range(len(features)):
            x=features[src] if (dst,src) not in overrides else overrides[(dst,src)]
            if x is None:continue
            w=weight[offsets[dst]:offsets[dst+1],offsets[src]:offsets[src+1]]
            v=F.conv1d(x,w,padding=mixer.conv.kernel_size[0]//2)
            total=v if total is None else total+v
        outputs.append(total)
    return tuple(outputs)


def paired(g,data,out,permutation_file,selected_predictions,batch=16):
    ex=g.modules_by_name()['bridge_0_exchange']
    assert ex.mode=='radon' and ex.compression in ('fixed_svd_channel','fixed_random_orthogonal_channel')
    assert ex.keys==['cfp_stage3','oct_stage3']
    with np.load(permutation_file,allow_pickle=False) as z:perms=z['permutations'];expected_ids=z['ids']
    with np.load(selected_predictions,allow_pickle=False) as z:reference={k:z[k].copy() for k in ['ids','y','cfp','oct']}
    full={b:[] for b in g.branches};xs=[[],[]];zs=[[],[]];ids=[];ys=[];device=next(g.graph.parameters()).device
    with read_only(g),torch.no_grad():
        for c,o,y,keys in loader(data,batch,0):
            logits,_=g.forward(c.to(device),o.to(device),y.to(device));ids.extend(keys);ys.extend(y.tolist())
            for b in g.branches:full[b].append(logits[b].softmax(1).cpu())
            for i,x in enumerate(ex.latest_inputs):
                xs[i].append(x.cpu());zs[i].append(ex.projectors[i](ex.channel_bases[i].encode(x)).cpu())
        xs=[torch.cat(v) for v in xs];zs=[torch.cat(v) for v in zs];ys=np.asarray(ys);ids=np.asarray(ids)
        full={b:torch.cat(v).numpy() for b,v in full.items()};n=len(ys)
        assert np.array_equal(ids,expected_ids) and np.array_equal(ids,reference['ids']) and np.array_equal(ys,reference['y'])
        assert perms.shape==(20,n) and all(np.array_equal(np.sort(p),np.arange(n)) and np.all(p!=np.arange(n)) for p in perms)
        for b in full:assert np.allclose(full[b],reference[b],rtol=1e-5,atol=1e-6)
        errors=[];predictions=[];conditions=[];summary=[]
        schedule=[('paired',None),('disable_oct_to_cfp',None),('disable_cfp_to_oct',None),('disable_both',None)]
        schedule += [(condition,i) for condition in ['shuffle_oct_to_cfp','shuffle_cfp_to_oct','shuffle_both'] for i in range(20)]
        def eyes(indices):return np.stack([2*indices,2*indices+1],1).reshape(-1)
        for condition,repeat in schedule:
            scores={b:[] for b in g.branches}
            for start in range(0,n,batch):
                idx=np.arange(start,min(start+batch,n));features=[x[eyes(idx)].to(device) for x in xs];overrides={}
                directions=[]
                if 'oct_to_cfp' in condition or condition.endswith('both'):directions.append((0,1))
                if 'cfp_to_oct' in condition or condition.endswith('both'):directions.append((1,0))
                for dst,src in directions:
                    if condition.startswith('disable'):overrides[(dst,src)]=None
                    else:
                        p=perms[repeat] if dst==0 else np.argsort(perms[repeat])
                        overrides[(dst,src)]=zs[src][eyes(p[idx])].to(device)
                hook=None
                if condition!='paired':hook=ex.mixer.register_forward_hook(lambda module,inputs,output:block_messages(module,inputs,overrides))
                try:got=cached_forward(g,features,torch.as_tensor(ys[idx],device=device))
                finally:
                    if hook is not None:hook.remove()
                for b in g.branches:scores[b].append(got[b].cpu().numpy())
            scores={b:np.concatenate(v) for b,v in scores.items()}
            if condition=='paired':
                for b in scores:
                    error=float(np.max(np.abs(scores[b]-full[b])));errors.append(error);assert np.allclose(scores[b],full[b],rtol=1e-5,atol=1e-6)
            record={'condition':condition,'repeat':repeat,'branches':{}}
            for b in g.branches:
                record['branches'][b]={'metrics':classification_metrics(ys,scores[b]),'mean_absolute_probability_change':float(np.abs(scores[b]-full[b]).mean()),'maximum_absolute_probability_change':float(np.abs(scores[b]-full[b]).max())}
            record['mean_branch_f1']=np.mean([record['branches'][b]['metrics']['macro_f1'] for b in g.branches]).item()
            summary.append(record);conditions.append([condition,repeat]);predictions.append(np.stack([scores[b] for b in g.branches]))
            write_json(out/'progress.json',{'completed_conditions':len(summary),'total_conditions':64})
        release_forward_graph(g)
    np.savez_compressed(out/'private_perturbation_predictions.npz',ids=ids,y=ys,conditions=np.asarray(json.dumps(conditions)),probabilities=np.stack(predictions))
    aggregates={}
    for condition in ['shuffle_oct_to_cfp','shuffle_cfp_to_oct','shuffle_both']:
        rows=[r for r in summary if r['condition']==condition];aggregates[condition]={}
        for b in g.branches:
            aggregates[condition][b]={}
            for metric in ['macro_f1','mean_absolute_probability_change']:
                v=[r['branches'][b]['metrics'][metric] if metric=='macro_f1' else r['branches'][b][metric] for r in rows]
                aggregates[condition][b][metric]={'mean':float(np.mean(v)),'sample_sd':float(np.std(v,ddof=1)),'min':float(min(v)),'max':float(max(v))}
    return {'conditions':summary,'permutation_aggregates':aggregates,'permutation_sha256':file_sha(permutation_file),'participants':n,'permutations':20,'cached_full_forward_max_probability_error':max(errors),'state_parameters_bn_gradients_rng_preserved':True}


def gpu_load():
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines()
    device=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    uuid=subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    return [int(line.split(',')[0]) for line in lines if line.split(',')[-1].strip()==uuid and int(line.split(',')[0])!=os.getpid()]


def latency(g,data):
    from torch.utils.data import Subset
    ids=sorted(range(len(data)),key=lambda i:data.rows[i]['order'])[:16];device=next(g.graph.parameters()).device
    c,o,y,_=next(iter(loader(Subset(data,ids),16,0)));c=c.to(device);o=o.to(device);y=y.to(device);samples=[];others=[]
    with read_only(g),torch.no_grad():
        for _ in range(10):g.forward(c,o,y)
        torch.cuda.synchronize()
        for _ in range(50):
            others.extend(gpu_load());torch.cuda.synchronize();tick=time.perf_counter();g.forward(c,o,y);torch.cuda.synchronize();samples.append(1000*(time.perf_counter()-tick))
        release_forward_graph(g)
    return {'milliseconds':samples,'median_ms':float(np.median(samples)),'q25_ms':float(np.quantile(samples,.25)),'q75_ms':float(np.quantile(samples,.75)),'warmup':10,'repetitions':50,'batch':16,'device':torch.cuda.get_device_name(0),'gpu_index':os.environ.get('CUDA_VISIBLE_DEVICES'), 'other_gpu_processes_detected':bool(others),'timing_interfered':bool(others),'sample_ids_sha256':hashlib.sha256(json.dumps([data.rows[i]['order'] for i in ids],separators=(',',':')).encode()).hexdigest(),'state_parameters_bn_gradients_rng_preserved':True}


def main(cfg,out,data_path):
    tick=time.monotonic();torch.set_num_threads(3);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from mhd_models.scheduling.gpu_budget import configure_allocator
    configure_allocator(0)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    trial=Path(cfg['trial_directory']);config=json.loads((trial/'configuration.json').read_text());checkpoint=trial/cfg.get('selected_file','selected.pt');assert file_sha(checkpoint)==cfg['selected_sha256']
    g=PilotGraph(seed=config['seed'],bridge_configs=config['bridges'],device='cuda');saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert set(saved['model'])==set(g.modules_by_name())
    for key,module in g.modules_by_name().items():module.load_state_dict(saved['model'][key],strict=True)
    del saved
    if cfg['analysis']=='pairing':result=paired(g,PairedDataset(data_path,'validation',224),out,cfg['permutation_file'],trial/'selected_predictions.npz')
    elif cfg['analysis']=='latency':result=latency(g,PairedDataset(data_path,'train',224))
    else:raise ValueError(cfg['analysis'])
    write_json(out/'summary.json',dict(state='complete',passed=True,analysis=cfg['analysis'],selected_sha256=cfg['selected_sha256'],trial_directory=str(trial),seconds=time.monotonic()-tick,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,result=result,test_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--data',required=True);a=p.parse_args()
    try:main(json.loads(Path(a.config).read_text()),a.output,a.data)
    except BaseException:
        import traceback
        write_json(Path(a.output)/'failure.json',{'state':'failed','error':traceback.format_exc()});raise
