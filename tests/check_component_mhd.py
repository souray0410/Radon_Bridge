"""Synthetic-input four-state checks on real accepted parallel-host checkpoints."""
import json,gc
from pathlib import Path
import torch
from radonbridge.artifacts import SOURCE,STUDY,resolve
from radonbridge.component_ablation import component_switch,STATES
from radonbridge.diagnostics import read_only,release_forward_graph
from radonbridge.model import PilotGraph

torch.set_num_threads(3)
rows=json.loads((SOURCE/'runs'/STUDY/'branch_only/manifest.json').read_text())['rows']
checks=[]
for host in ('mmtm_hidden256','attention_d256'):
    row=next(r for r in rows if r['category']=='augmentation' and r['state']=='accepted' and r['structure']['host']==host and r['structure']['arm']=='radon')
    saved=torch.load(resolve(Path(row['directory'])/'selected.pt'),map_location='cpu',weights_only=False);cfg=saved['configuration']
    g=PilotGraph(bridge_configs=cfg['bridges'],seed=cfg['seed'],device='cpu');g.load_complete_state(saved['model'])
    c=torch.randn(1,2,3,224,224);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([1]);pred=[]
    node_ids={k:g.by_name[k].id for k in ('cfp_stage3','oct_stage3')}
    with read_only(g),torch.no_grad():
        for on_host,on_new in STATES:
            with component_switch(g,on_host,on_new):
                z,_=g.forward(c,o,y);pred.append({k:v.clone() for k,v in z.items()});release_forward_graph(g)
        z,_=g.forward(c,o,y);assert all(torch.equal(z[k],pred[0][k]) for k in z)
        assert node_ids=={k:g.by_name[k].id for k in node_ids}
        release_forward_graph(g)
    state=g.save_state();del g;gc.collect()
    native=PilotGraph(bridge_configs=[],seed=cfg['seed'],device='cpu')
    native.load_native_state({k:v for k,v in state.items() if k in native.definition.native_checkpoint_modules})
    with read_only(native),torch.no_grad():
        z,_=native.forward(c,o,y);assert all(torch.allclose(z[k],pred[-1][k],atol=1e-6,rtol=1e-5) for k in z)
        release_forward_graph(native)
    checks.append(host);del native,state,saved;gc.collect()
print(json.dumps(dict(passed=True,hosts=checks,states=4,full_replay_exact=True,both_off_matches_updated_native=True,original_node_ids=True)))
