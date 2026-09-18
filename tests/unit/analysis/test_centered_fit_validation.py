import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.validate_centered_fit import recompute_fingerprint, validate


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def build_fixture(tmp_path):
    tmp_path.mkdir(parents=True,exist_ok=True)
    data=tmp_path/'data';data.mkdir();(data/'audit.json').write_text('audit');(data/'selected.csv').write_text('selected')
    parents={'cfp':{'path':'/archive/cfp.pt','sha256':'a'},'oct':{'path':'/archive/oct.pt','sha256':'b'}}
    nodes=['cfp_stage3','oct_stage3'];old={};native='native-sha'
    ids=[f'id-{i}' for i in range(1264)]
    import hashlib
    old_ids=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    for i,key in enumerate(nodes):
        q=np.eye(4,dtype=np.float64);second=np.diag(np.array([5.,4.,3.,2.])+i)
        meta={'version':'train_channel_second_moment_eigh_v1','source_key':key,'seed':3416,'channels':4,
              'sampled_channel_vectors':10+i,'centered':False,'fit_split':'train','test_used':False,
              'provenance':{'parent_checkpoints':parents,'initial_native_sha256':native,'participant_ids_sha256':old_ids}}
        path=tmp_path/f'old_{i}.npz';np.savez(path,q=q,eigenvalues=np.diag(second),second_moment=second,metadata=json.dumps(meta))
        old[key]={'path':str(path),'sha256':sha(path)}
    cfg={'seed':3416,'parent_checkpoints':parents,'uncentered_basis_files':old,'nodes':nodes,'microbatch':16,
         'source_commit':'source','fit_centered':True,'fit_domain':'native stage3 channel features; all eyes and spatial positions equally weighted'}
    root=tmp_path/'run';fit=root/'basis_fit';bases=fit/'bases';bases.mkdir(parents=True)
    (root/'basis_fit_config.json').write_text(json.dumps(cfg))
    (root/'protocol.json').write_text(json.dumps({'study_kind':'centered_basis','test_used':False,'data':str(data)}))
    fingerprint,payload=recompute_fingerprint(cfg,data,native)
    ids_digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    moments={};sums={};counts={};new={}
    for i,key in enumerate(nodes):
        count=10+i;mean=np.array([.2,.1,-.1,.05],dtype=np.float64)
        covariance=np.diag(np.array([4.,3.,2.,1.])+i)
        second=covariance+np.outer(mean,mean);moment=torch.from_numpy(second*count);channel_sum=torch.from_numpy(mean*count)
        moments[key]=moment;sums[key]=channel_sum;counts[key]=count
        eigen=np.diag(covariance).copy();q=np.eye(4,dtype=np.float64)
        pivots=np.argmax(np.abs(q),axis=0);assert np.all(q[pivots,np.arange(4)]>=0)
        energy=np.sum(q*(second@q),axis=0)
        meta={'version':'train_channel_centered_covariance_eigh_v1','source_key':key,'seed':3416,'channels':4,
              'sampled_channel_vectors':count,'centered':True,'runtime_centering':False,'fit_split':'train','test_used':False,
              'full_master_sha256':tensor_sha(q),'mean_sha256':tensor_sha(mean),
              'provenance':{'statistics_fingerprint':fingerprint}}
        path=bases/f'{key}.npz';new[key]={'path':str(path),'sha256':None}
        np.savez(path,q=q,eigenvalues=eigen,energies=energy,mean=mean,second_moment=second,covariance=covariance,metadata=json.dumps(meta))
        new[key]['sha256']=sha(path)
    state={'schema':'train_channel_sufficient_statistics_state_v1','fingerprint':fingerprint,'offset':1264,
           'moments':moments,'sums':sums,'counts':counts,'ids':ids}
    state_path=fit/'sufficient_statistics.pt';torch.save(state,state_path);state_sha=sha(state_path)
    for key in nodes:
        p=Path(new[key]['path'])
        with np.load(p,allow_pickle=False) as z:
            arrays={n:z[n].copy() for n in z.files if n!='metadata'};meta=json.loads(str(z['metadata']))
        meta['provenance']['sufficient_statistics_sha256']=state_sha
        np.savez(p,**arrays,metadata=json.dumps(meta));new[key]['sha256']=sha(p)
    summary={'state':'complete','passed':True,'test_used':False,'bases':new,
             'provenance':{'participant_ids_sha256':ids_digest,'statistics_fingerprint':fingerprint,
                           'statistics_fingerprint_payload':payload,'sufficient_statistics_sha256':state_sha,
                           'initial_native_sha256':native,'parent_checkpoints':parents}}
    (fit/'summary.json').write_text(json.dumps(summary))
    # Rewrite basis metadata after final state SHA only; summary refs already updated above.
    return root


def sync_state_sha(root):
    state=root/'basis_fit/sufficient_statistics.pt';summary_path=root/'basis_fit/summary.json';summary=json.loads(summary_path.read_text())
    digest=sha(state);summary['provenance']['sufficient_statistics_sha256']=digest
    for ref in summary['bases'].values():
        p=Path(ref['path'])
        with np.load(p,allow_pickle=False) as z:
            arrays={n:z[n].copy() for n in z.files if n!='metadata'};meta=json.loads(str(z['metadata']))
        meta['provenance']['sufficient_statistics_sha256']=digest
        np.savez(p,**arrays,metadata=json.dumps(meta));ref['sha256']=sha(p)
    summary_path.write_text(json.dumps(summary))


def test_independent_centered_numeric_acceptance(tmp_path):
    root=build_fixture(tmp_path);value=validate(root)
    assert value['passed'] and value['participants']==1264
    assert max(v['orthogonality_max_abs'] for v in value['measured'].values())==0


@pytest.mark.parametrize('field', ['sums','moments','counts'])
def test_wrong_sufficient_statistics_are_rejected_even_with_updated_sha(tmp_path,field):
    root=build_fixture(tmp_path);state_path=root/'basis_fit/sufficient_statistics.pt'
    state=torch.load(state_path,map_location='cpu',weights_only=False)
    key='cfp_stage3'
    if field=='sums':state[field][key]=state[field][key].clone();state[field][key][0]+=0.5
    elif field=='moments':state[field][key]=state[field][key].clone();state[field][key][0,0]+=0.5
    else:state[field][key]+=1
    torch.save(state,state_path);sync_state_sha(root)
    with pytest.raises(ValueError):
        validate(root)


def test_synchronized_fake_fingerprint_is_rejected(tmp_path):
    root=build_fixture(tmp_path);state_path=root/'basis_fit/sufficient_statistics.pt';summary_path=root/'basis_fit/summary.json'
    state=torch.load(state_path,map_location='cpu',weights_only=False);state['fingerprint']='fake';torch.save(state,state_path)
    summary=json.loads(summary_path.read_text());summary['provenance']['statistics_fingerprint']='fake'
    summary_path.write_text(json.dumps(summary));sync_state_sha(root)
    with pytest.raises(ValueError,match='fingerprint'):
        validate(root)


def test_corrupt_q_eigen_or_energy_is_rejected(tmp_path):
    for field in ('q','eigenvalues','energies'):
        root=build_fixture(tmp_path/field);summary=json.loads((root/'basis_fit/summary.json').read_text())
        ref=summary['bases']['cfp_stage3'];p=Path(ref['path'])
        with np.load(p,allow_pickle=False) as z:
            arrays={n:z[n].copy() for n in z.files}
        if field=='q':
            arrays['q'][0,0]+=0.01
            meta=json.loads(str(arrays['metadata']));meta['full_master_sha256']=tensor_sha(arrays['q']);arrays['metadata']=json.dumps(meta)
        else:arrays[field][0]+=0.25
        np.savez(p,**arrays);ref['sha256']=sha(p);(root/'basis_fit/summary.json').write_text(json.dumps(summary))
        with pytest.raises(ValueError):
            validate(root)
