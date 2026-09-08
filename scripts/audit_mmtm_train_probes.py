"""Read-only MMTM stopping-state probes on a fixed training subset, GPU 0 only.

Not new training: no optimizer construction/update, no test/development forward.
"""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from radonbridge.artifacts import relocate,sha256
from radonbridge.data import PairedDataset
from radonbridge.model import PilotGraph
from radonbridge.diagnostics import read_only,release_forward_graph,state_hash

def read(p):return json.loads(Path(p).read_text())
def stats(x):
    x=torch.cat(x).double();return dict(mean=float(x.mean()),sd=float(x.std()),min=float(x.min()),max=float(x.max()),q01=float(x.quantile(.01)),q99=float(x.quantile(.99)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert read('/data/mengh/RadonBridge/gpu_allocation.json')['gpu_indices']==[0]
    a.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    lock=a.root/'unified_test_lock_v6';data=PairedDataset(read(lock/'candidate_lock.json')['development_data_directory'],'train',224)
    ii=sorted(range(len(data)),key=lambda i:data.rows[i]['order'])[:128]
    probe_sha=hashlib.sha256(json.dumps([data.rows[i]['order'] for i in ii],separators=(',',':')).encode()).hexdigest()
    jobs=[]
    for j in read(lock/'jobs.json'):
        ref=next((r for r in j['model']['reference_ids'] if r.startswith('current_direct_and_host:branch_mmtm_') and '_augment_' not in r),None)
        if j['kind']=='model' and ref:jobs.append((ref,j['model']))
    assert len(jobs)==24
    allrows=[];start=time.monotonic()
    for ref,m in sorted(jobs):
        cfg=relocate(m['configuration']);parent=Path(m['checkpoint_path']).parent
        assert sha256(m['checkpoint_path'])==m['checkpoint_sha256']
        selected=torch.load(m['checkpoint_path'],map_location='cpu',weights_only=False)
        last_path=parent/'last.pt';last=torch.load(last_path,map_location='cpu',weights_only=False)
        g=PilotGraph(seed=cfg['seed'],bridge_configs=cfg['bridges'],device='cuda')
        g.load_complete_state(last['model']);g.graph.eval();ex=g.modules_by_name()['bridge_0_exchange']
        w={};bn={}
        for name,sd in last['model'].items():
            if name not in selected['model']:raise ValueError('State module mismatch')
            old=selected['model'][name];diff=0.;base=0.
            for k,v in sd.items():
                if not v.is_floating_point():continue
                if 'running_' in k:bn[name+'/'+k]=float((v-old[k]).double().norm())
                else:diff+=float((v-old[k]).double().square().sum());base+=float(old[k].double().square().sum())
            w[name]=dict(delta_l2=diff**.5,initial_l2=base**.5,relative_delta= (diff/base)**.5 if base else None)
        # Saved optimizer moments prove updates occurred; no optimizer is instantiated.
        moments={}
        if 'optimizer' in last:
            for group in last['optimizer']['param_groups']:
                vals=[last['optimizer']['state'].get(k,{}) for k in group['params']]
                moments[group.get('name','unnamed')]=dict(steps=sorted({int(v['step']) for v in vals if 'step' in v}),exp_avg_l2=sum(float(v['exp_avg'].double().square().sum()) for v in vals if 'exp_avg' in v)**.5)
        gates=[[],[]];hidden=[];en=np.zeros((2,2));grads={}
        before=state_hash(g)
        with read_only(g):
            for idx,(c,o,y,_) in enumerate(DataLoader(Subset(data,ii),batch_size=16,shuffle=False,num_workers=0)):
                c,o,y=c.cuda(),o.cuda(),y.cuda()
                with torch.no_grad():
                    g.forward(c,o,y);xs=ex.latest_inputs
                    z=torch.relu(ex.squeeze(torch.cat([v.flatten(2).mean(-1) for v in xs],1)));hidden.append((z==0).float().flatten().cpu())
                    for k,(v,l) in enumerate(zip(xs,ex.excite)):
                        gate=2*torch.sigmoid(l(z));gates[k].append(gate.flatten().cpu())
                        en[k,0]+=float(v.double().square().sum());en[k,1]+=float(ex.latest_deltas[k].double().square().sum())
                release_forward_graph(g)
                # Fixed representative subset: first 16 training participants, seed3416/lr6e-5, all widths.
                if idx==0 and cfg['seed']==3416 and cfg['backbone_lr']==6e-5:
                    # Only bridge derivatives are requested: disabling native parameter
                    # gradients avoids storing upstream activations without altering
                    # the mathematical derivative of the CE with respect to the bridge.
                    flags={v:v.requires_grad for v in g.graph.parameters()}
                    try:
                        for v in g.graph.parameters():v.requires_grad_(False)
                        for v in ex.parameters():v.requires_grad_(True)
                        for state,checkpoint in [('stopping',last),('selected_initial',selected)]:
                            g.load_complete_state(checkpoint['model']);g.graph.eval();g.forward(c,o,y)
                            pars=list(ex.named_parameters());ce=g.by_name['cfp_loss'].feature_message.current_state+g.by_name['oct_loss'].feature_message.current_state
                            gg=torch.autograd.grad(ce,[v for _,v in pars],allow_unused=True)
                            grads[state]={n:None if v is None else float(v.norm()) for (n,_),v in zip(pars,gg)}
                            release_forward_graph(g)
                    finally:
                        g.load_complete_state(last['model'])
                        for v,flag in flags.items():v.requires_grad_(flag)
        assert state_hash(g)==before
        del last,selected
        gate_report=[]
        for k in range(2):
            arr=torch.cat(gates[k]);gate_report.append(dict(branch=['cfp','oct'][k],**stats(gates[k]),fraction_sigmoid_below01_or_above99=float(((arr<.02)|(arr>1.98)).float().mean()),residual_over_feature_l2=float((en[k,1]/en[k,0])**.5)))
        row=dict(reference=ref,seed=cfg['seed'],backbone_lr=cfg['backbone_lr'],hidden_dimension=ex.squeeze.out_features,selected_sha256=m['checkpoint_sha256'],last_sha256=sha256(last_path),gate=gate_report,hidden_relu_zero_fraction=float(torch.cat(hidden).mean()),module_weight_changes=w,bn_changed_buffers=sum(v>0 for v in bn.values()),optimizer_moments=moments,gradient_probe=grads,parameters_bn_gradients_rng_preserved=True)
        allrows.append(row)
        (a.output/'mmtm_probe_results.json').write_text(json.dumps(allrows,indent=2))
        print(json.dumps(dict(accepted=len(allrows),total=24,hidden=row['hidden_dimension'],gate_max=[x['max'] for x in gate_report])),flush=True)
        release_forward_graph(g);del g,ex,c,o,y;gc.collect();torch.cuda.empty_cache()
    peak=torch.cuda.max_memory_reserved()/1024**2;assert peak<=10240
    (a.output/'status.json').write_text(json.dumps(dict(state='complete',checkpoints=24,train_probe_participants=128,gradient_probe_participants=16,gradient_probe_configurations=4,probe_sha256=probe_sha,gpu=0,peak_reserved_mib=peak,seconds=time.monotonic()-start,no_training=True,no_test_forward=True,no_development_forward=True,read_only=True),indent=2))

if __name__=='__main__':main()
