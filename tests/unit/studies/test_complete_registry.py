import copy
import json
import pytest
from radon_bridge.studies.complete_matrix import all_positions, write_protocol, groups
from radon_bridge.studies.complete_registry import Registry, select_public


def test_positions_and_restart_preserve_progress(tmp_path):
    rows=list(all_positions())
    assert len(rows)==92526 and len({r['id'] for r in rows})==92526
    assert sum(r['phase']=='tuning' for r in rows)==1536
    assert sum(r['phase']=='selected' for r in rows)==10368
    assert sum(r['phase']=='selected_host' for r in rows)==15552
    write_protocol(tmp_path/'protocol')
    progress=tmp_path/'protocol/status.json';progress.write_text('{"accepted": 7}')
    write_protocol(tmp_path/'protocol')
    assert json.loads(progress.read_text())=={'accepted':7}
    registry=Registry(tmp_path/'ledger.sqlite');registry.register();registry.register()
    assert registry.counts()==dict(positions=92526,bound_positions=0,executions={})


def test_search_cannot_select_partial_or_failed_candidates():
    refs=[g['id'] for g in groups() if g['tuning_reference'] and len(g['sources'])==6]
    rows=[dict(group=g,candidate=i,seed=3416,family='mmtm',test_access=False,state='accepted',plateau=True,
               receipt_sha256='fixture',mean_macro_f1=.7,arithmetic_cost=100+i,parameters=100+i)
          for g in refs for i in range(32)]
    assert select_public('mmtm',6,rows)['candidate']==0
    with pytest.raises(ValueError,match='not complete'):select_public('mmtm',6,rows[:-1])
    bad=copy.deepcopy(rows);bad[0]['state']='failed'
    with pytest.raises(ValueError,match='Unresolved'):select_public('mmtm',6,bad)
    bad=copy.deepcopy(rows);bad[0]['test_access']=True
    with pytest.raises(ValueError):select_public('mmtm',6,bad)
