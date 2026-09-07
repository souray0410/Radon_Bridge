"""Allow audited scheduler-only upgrades while preserving scientific locks."""
import shutil
from scripts.geometry_evidence import read,write
from radonbridge.artifacts import sha256

ALLOWED={'scripts/run_integer_experiment.py','scripts/rolling_dispatch.py','scripts/gpu_allocation.py',
         'scripts/set_gpu_allocation.py','scripts/runtime_revision.py','scripts/run_depth_study.py',
         'tests/check_gpu_allocation.py','tests/check_rolling_dispatch.py','tests/check_scheduler.py'}

def revise(root,new,resume_from):
    path=root/'protocol.json';old=read(path)
    if old==new:return
    assert resume_from==old['source_commit'],'Explicit previous scheduler version required'
    skip={'source_commit','accepted_source_hashes'}
    assert {k:v for k,v in old.items() if k not in skip}=={k:v for k,v in new.items() if k not in skip},'Scientific protocol changed'
    changed={k for k in set(old['accepted_source_hashes'])|set(new['accepted_source_hashes']) if old['accepted_source_hashes'].get(k)!=new['accepted_source_hashes'].get(k)}
    assert changed and changed<=ALLOWED,changed
    target=root/'protocol_revisions'/f'scheduler_{resume_from}.json';target.parent.mkdir(exist_ok=True)
    if target.exists():assert sha256(target)==sha256(path)
    else:shutil.copyfile(path,target)
    record=dict(previous_commit=resume_from,current_commit=new['source_commit'],changed_files=sorted(changed),
        previous_protocol_sha256=sha256(target),scientific_protocol_changed=False,
        candidate_lock_sha256=sha256(root/'candidate_lock.json') if (root/'candidate_lock.json').exists() else None)
    write(root/'runtime_revision.json',record);write(path,new)
