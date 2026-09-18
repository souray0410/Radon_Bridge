import json
from pathlib import Path
import pytest
import torch
from radon_bridge.studies.cohort_augmentation import migrate,load_host
from radon_bridge.studies.cohort_case import sha


def test_migration_and_exact_host_identity(tmp_path):
    src=tmp_path/'old';src.mkdir()
    cfg=dict(seed=3416,parents={'cfp':{'sha256':'a'},'oct':{'sha256':'b'}},bridges=[dict(family='mmtm',nodes=['cfp_stage3','oct_stage3'])])
    torch.save(dict(configuration=cfg,model={'native':{'x':torch.ones(1)}}),src/'best.pt')
    (src/'selected_predictions.npz').write_bytes(b'fixture')
    (src/'accepted.json').write_text(json.dumps(dict(converged_by_policy=True,test_used=False,configuration=cfg,best_epoch=0,files={n:sha(src/n) for n in ('best.pt','selected_predictions.npz')})))
    dest=tmp_path/'canonical';migrate(src,dest)
    new=dict(cfg,augmentation_host=dict(path=str(dest/'manifest.json'),sha256=sha(dest/'manifest.json')))
    class Graph:
        def load_complete_state(self,state,allow_new_bridge):self.add=allow_new_bridge;assert torch.equal(state['native']['x'],torch.ones(1))
    g=Graph();load_host(new,g);assert not g.add
    new['bridges']=[*cfg['bridges'],dict(parallel_to=0)];load_host(new,g);assert g.add
    new['seed']=3417
    with pytest.raises(ValueError,match='Unmatched'):load_host(new,g)
    new['seed']=3416;(src/'selected_predictions.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='predictions changed'):load_host(new,g)
