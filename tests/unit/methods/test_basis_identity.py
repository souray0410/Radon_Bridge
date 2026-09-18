import torch
from radon_bridge.methods.basis import (
    archive_incomplete_basis_directory,
    load_statistics_state,
    same_parent_identity,
    save_statistics_state,
    statistics_fingerprint,
)


def test_parent_identity_allows_relocation_but_not_weight_change():
    old={'cfp':{'path':'/data/old/cfp.pt','sha256':'a'},'oct':{'path':'/data/old/oct.pt','sha256':'b'}}
    relocated={'cfp':{'path':'/backup/archive/cfp.pt','sha256':'a'},'oct':{'path':'/backup/archive/oct.pt','sha256':'b'}}
    changed={'cfp':{'path':'/backup/archive/cfp.pt','sha256':'different'},'oct':{'path':'/backup/archive/oct.pt','sha256':'b'}}
    assert same_parent_identity(old,relocated)
    assert not same_parent_identity(old,changed)
    assert not same_parent_identity(old,{'cfp':relocated['cfp']})


def test_sufficient_statistics_cache_is_identity_bound_and_resumable(tmp_path):
    data=tmp_path/'data';data.mkdir();(data/'audit.json').write_text('audit');(data/'selected.csv').write_text('selected')
    cfg={'seed':3416,'parent_checkpoints':{'cfp':{'path':'old','sha256':'a'},'oct':{'path':'old2','sha256':'b'}},
        'uncentered_basis_files':{'cfp_stage3':{'path':'c','sha256':'c'},'oct_stage3':{'path':'o','sha256':'d'}},
        'nodes':['cfp_stage3','oct_stage3'],'microbatch':16,'source_commit':'source','fit_centered':True}
    fingerprint,_=statistics_fingerprint(cfg,data,'native')
    moments={k:torch.eye(2,dtype=torch.float64) for k in cfg['nodes']}
    sums={k:torch.ones(2,dtype=torch.float64) for k in cfg['nodes']};counts={k:10 for k in cfg['nodes']}
    path=tmp_path/'stats.pt';save_statistics_state(path,fingerprint,1,moments,sums,counts,['id'])
    value=load_statistics_state(path,fingerprint,cfg['nodes'],True,1264)
    assert value['offset']==1 and value['ids']==['id']
    import pytest
    with pytest.raises(ValueError,match='identity changed'):load_statistics_state(path,'wrong',cfg['nodes'],True,1264)
    relocated=dict(cfg,parent_checkpoints={'cfp':{'path':'new','sha256':'a'},'oct':{'path':'new2','sha256':'b'}})
    assert statistics_fingerprint(relocated,data,'native')[0]==fingerprint
    changed=dict(cfg,parent_checkpoints={'cfp':{'path':'new','sha256':'changed'},'oct':{'path':'new2','sha256':'b'}})
    assert statistics_fingerprint(changed,data,'native')[0]!=fingerprint


def test_incomplete_basis_directory_is_preserved_as_attempt(tmp_path):
    bases=tmp_path/'bases';bases.mkdir();(bases/'partial.npz').write_bytes(b'partial')
    archived=archive_incomplete_basis_directory(tmp_path)
    assert archived.is_dir() and (archived/'partial.npz').read_bytes()==b'partial' and not bases.exists()
