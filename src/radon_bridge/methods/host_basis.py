"""Fit one frozen SVD basis to an intact selected host's pre-write TRAIN features."""
import argparse,json,time
from pathlib import Path
import torch
from radon_bridge.runtime.artifacts import relocate, sha256
from radon_bridge.models.model import PilotGraph
from radon_bridge.data.dataset import PairedDataset
from radon_bridge.training.trainer import loader, write_json
from radon_bridge.analysis.diagnostics import read_only, release_forward_graph
from radon_bridge.methods.basis import save_basis

def main(cfg,out,data):
    start=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise RuntimeError('Basis fit already complete')
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    from mhd_models.scheduling.gpu_budget import configure_allocator
    configure_allocator(0)
    cfg=relocate(cfg);ref=cfg['host_checkpoint'];assert sha256(ref['path'])==ref['sha256']
    saved=torch.load(ref['path'],map_location='cpu',weights_only=False);c=relocate(saved['configuration'])
    g=PilotGraph(seed=c['seed'],bridge_configs=c['bridges'],task_fusion=c.get('task_fusion'),device='cuda')
    g.load_complete_state(saved['model']);del saved
    train=PairedDataset(data,'train',224);assert len(train)==1264
    exchange=g.modules_by_name()['bridge_0_exchange'];keys=exchange.keys
    moments={k:torch.zeros(256,256,dtype=torch.float64) for k in keys};counts={k:0 for k in keys};orders=[]
    with read_only(g),torch.no_grad():
        for cfp,oct_,target,ids in loader(train,16,c['seed']):
            g.forward(cfp.cuda(),oct_.cuda(),target.cuda());orders.extend(list(ids))
            for key,x in zip(keys,exchange.latest_inputs):
                f=x.movedim(1,0).reshape(256,-1).cpu().double()
                moments[key].add_(f@f.T);counts[key]+=f.shape[1]
            release_forward_graph(g)
    provenance=dict(host_checkpoint=ref,configuration=c,participant_order_sha256=__import__('hashlib').sha256(json.dumps(orders,separators=(',',':')).encode()).hexdigest(),
                    training_participants=1264,feature_position='intact host stage3 before communication write-back',state_preserved=True)
    bases={k:save_basis(moments[k],counts[k],out/'bases',k,c['seed'],provenance) for k in keys}
    write_json(out/'summary.json',dict(state='complete',passed=True,basis_files=bases,provenance=provenance,
                                     seconds=time.monotonic()-start,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,test_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--data',required=True);a=p.parse_args()
    main(json.loads(Path(a.config).read_text()),a.output,a.data)
