"""Operational GPU selection, separate from immutable scientific configurations."""
import hashlib,json,os,time
from pathlib import Path

DEFAULT_PATH=Path('/data/mengh/RadonBridge/gpu_allocation.json')

def validate(indices,inventory=None):
    if not isinstance(indices,list) or any(type(i)!=int or i<0 for i in indices) or len(set(indices))!=len(indices):
        raise ValueError('GPU indices must be distinct nonnegative integers; [] pauses dispatch')
    if inventory is not None and any(i not in inventory for i in indices):
        raise ValueError('Selected GPU not present in physical device inventory')
    return list(indices)

class Allocation:
    def __init__(self,path,default,required=False):
        self.path=Path(path);self.default=validate(default);self.required=required;self.seen=False;self.last=None;self.current={}

    def refresh(self,inventory):
        try:
            if self.path.exists():
                self.seen=True;raw=self.path.read_bytes();d=json.loads(raw)
                if set(d)!={'schema','gpu_indices','updated_at','behavior'} or d['schema']!='gpu_allocation_v1' or d['behavior']!='finish_active':
                    raise ValueError('Invalid allocation schema or unsupported behavior')
                selected=validate(d['gpu_indices'],inventory);origin='runtime_file';digest=hashlib.sha256(raw).hexdigest()
            elif self.seen or self.required:raise ValueError('Required allocation file missing; dispatch paused')
            else:selected=validate(self.default,inventory);origin='protocol_default';digest=None
            value=dict(selected_gpu_indices=selected,selected_gpu_uuids={str(i):inventory[i]['uuid'] for i in selected},
                       allocation_source=origin,allocation_sha256=digest,allocation_error=None)
        except (OSError,ValueError,TypeError,KeyError) as exc:
            value=dict(selected_gpu_indices=[],selected_gpu_uuids={},allocation_source='invalid_runtime_allocation',
                       allocation_sha256=None,allocation_error=str(exc))
        changed=value!=self.last;self.last=value;self.current=value
        return value,changed

def refresh_controller(controller,inventory):
    allocation=getattr(controller,'allocation',None)
    if allocation is None:return controller.gpus  # Legacy test fixtures/custom controllers.
    value,changed=allocation.refresh(inventory)
    controller.allocation_status=dict(value)
    if changed:
        from radonbridge.experiment import write_json
        controller.ledger.setdefault('allocation_events',[]).append(dict(observed_at=time.time(),**value,
            active_jobs=[dict(id=k,pid=v['process'].pid,gpu=v['gpu']) for k,v in controller.active.items()]))
        write_json(controller.root/'ledger.json',controller.ledger)
    return value['selected_gpu_indices']

def publish(path,indices,inventory):
    validate(indices,inventory);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(schema='gpu_allocation_v1',gpu_indices=indices,updated_at=time.time(),behavior='finish_active')
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
    return value

def project_worker(pid):
    """Recognize our worker command, without classifying LOOK by shared libraries."""
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return any(a==b'-m' and i+1<len(args) and args[i+1].startswith(b'radonbridge.') for i,a in enumerate(args))
    except OSError:return False

def memory_snapshot(raw,active,inventory,identify=project_worker):
    own={v['process'].pid for v in active.values()};indices={v['uuid']:k for k,v in inventory.items()}
    by_pid={};by_gpu={};project_processes={}
    for line in raw.splitlines():
        if not line.strip():continue
        pid,uuid,memory=[x.strip() for x in line.split(',')];pid=int(pid);memory=int(memory)
        by_pid[pid]=by_pid.get(pid,0)+memory
        if pid in own or identify(pid):
            gpu=indices[uuid];by_gpu[gpu]=by_gpu.get(gpu,0)+memory
            project_processes.setdefault(gpu,[]).append(pid)
    return by_pid,by_gpu,project_processes
