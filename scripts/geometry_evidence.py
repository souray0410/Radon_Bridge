"""Content-addressed historical acceptance and full completed-trial archiving."""
import json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
from radonbridge.artifacts import SOURCE,ARCHIVE,STUDY,resolve,sha256,relocate
from radonbridge.geometry_study import SEEDS,POLICY,semantic,fingerprint

FILES=('summary.json','configuration.json','selected.pt','selected_predictions.npz')
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)

def archive_path(path):
    p=Path(path)
    try:return ARCHIVE/'history'/p.relative_to(SOURCE)
    except ValueError:return p

def force_archive(value):
    if isinstance(value,dict):return {k:force_archive(v) for k,v in value.items()}
    if isinstance(value,list):return [force_archive(v) for v in value]
    if isinstance(value,str) and value.startswith(str(SOURCE)+'/'):return str(archive_path(value))
    return value

def dependencies():
    previous=ARCHIVE/'history/runs/2026_09_05_22_42_53/manifest.json'
    m=read(previous);parents={};bases={}
    for seed in SEEDS:
        row=next(r for r in m['rows'] if r['seed']==seed and r['arm']=='svd_radon' and r['rho']==.125)
        cfg=row['configuration'];parents[seed]=force_archive(cfg['parent_checkpoints']);bases[seed]=force_archive(cfg['bridges'][0]['basis_files'])
        for ref in list(parents[seed].values())+list(bases[seed].values()):assert sha256(ref['path'])==ref['sha256']
        from radonbridge.svd_basis import _load_basis,BASIS_VERSION
        for key,ref in bases[seed].items():
            _,_,metadata=_load_basis(ref['path'],ref['sha256'])
            assert metadata['version']==BASIS_VERSION and metadata['seed']==seed and metadata['source_key']==key
            assert {k:v['sha256'] for k,v in metadata['provenance']['parent_checkpoints'].items()}=={k:v['sha256'] for k,v in parents[seed].items()}
    return parents,bases

def scores(s,protocol):
    tasks=s['selected']['tasks'];a=tasks['cfp']['macro_f1'];b=tasks['oct']['macro_f1']
    return dict(cfp=a,oct=b,branch_mean=(a+b)/2,primary=tasks['fusion']['macro_f1'] if protocol=='fusion' else (a+b)/2)

def accept(path,cfg,protocol,expected=None):
    path=Path(path);s=read(path/'summary.json')
    assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau' and s['test_used'] is False
    assert semantic(s['configuration'])==semantic(cfg)==semantic(read(path/'configuration.json'))
    assert cfg['convergence']==POLICY and cfg['microbatch']==cfg['effective_batch']==16
    hashes={n:sha256(path/n) for n in FILES}
    if expected is not None:assert hashes==expected
    with np.load(path/'selected_predictions.npz',allow_pickle=False) as z:
        assert len(z['ids'])==len(set(z['ids']))==len(z['y'])==296
        assert np.bincount(z['y'],minlength=2).tolist()==[148,148]
        assert all(z[k].shape==(296,2) and np.isfinite(z[k]).all() for k in ('cfp','oct'))
        if protocol=='fusion':assert z['fusion'].shape==(296,2)
    return dict(directory=str(path),accepted_hashes=hashes,scores=scores(s,protocol),
                epochs=s['epochs_ran'],best_epoch=s['selection']['joint']['best_epoch'],seconds=s['seconds'],
                parameters=s['parameters'],peak_reserved_mib=s['peak_reserved_mib'],fingerprint=fingerprint(cfg))

def historical_index(wanted):
    """Match immutable configurations before reading performance; lexicographic first."""
    found={};excluded=[]
    for p in sorted((ARCHIVE/'history/runs').rglob('summary.json')):
        try:
            s=read(p);cfg=s.get('configuration',{})
            if not cfg.get('parent_checkpoints') or cfg.get('profile') or cfg.get('training_stage')!='communication':continue
            key=fingerprint(cfg)
            if key not in wanted:continue
            # Never replace an accepted result because a later duplicate scores better.
            if key in found:continue
            row=wanted[key]
            found[key]=accept(p.parent,row['configuration'],row['protocol'])
        except (ValueError,KeyError,AssertionError,OSError) as e:
            excluded.append(dict(path=str(p),reason=type(e).__name__+': '+str(e)))
    return found,excluded

def storage_check(work,peak_bytes,archive_peak_bytes=0):
    floor=100*1024**3
    status=dict(work_free=shutil.disk_usage(work).free,archive_free=shutil.disk_usage(ARCHIVE).free,
                work_required=floor+peak_bytes,archive_required=floor+archive_peak_bytes)
    if status['work_free']<status['work_required'] or status['archive_free']<status['archive_required']:
        raise RuntimeError('storage_needs_attention: '+json.dumps(status))
    return status

def archive_completed(path):
    """Preserve EVERY file and link in an accepted batch, including failed attempts."""
    path=Path(path);target=ARCHIVE/'current'/path.relative_to(SOURCE/'runs'/STUDY)
    target.parent.mkdir(parents=True,exist_ok=True)
    size=sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())
    storage_check(path,0,size)
    subprocess.run(['rsync','-aH','--',str(path)+'/',str(target)+'/'],check=True)
    records=[]
    for p in sorted(path.rglob('*')):
        q=target/p.relative_to(path)
        if p.is_symlink():
            assert q.is_symlink() and os.readlink(q)==os.readlink(p)
            # Complete batches may not rely on an unarchived external link.
            if not p.resolve().is_relative_to(path):raise RuntimeError('External batch symlink needs explicit mapping')
            records.append(dict(path=str(p.relative_to(path)),target=os.readlink(p)))
        elif p.is_file():
            digest=sha256(p);assert p.stat().st_size==q.stat().st_size and sha256(q)==digest
            records.append(dict(path=str(p.relative_to(path)),size=p.stat().st_size,sha256=digest))
    marker=path.parent/(path.name+'.archive.json')
    write(marker,dict(state='verified',original=str(path),archive=str(target),files=records))
    # The caller updates manifests to target first; retirement is a separate action.
    return target,marker
