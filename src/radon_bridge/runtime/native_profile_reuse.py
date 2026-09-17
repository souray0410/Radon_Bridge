"""Exact-spec reuse of closed whole-cohort resource evidence.

Only the expensive whole-input read/full-development phases are reusable. Every
new execution still runs the pinned GPU probe with 25 fresh worst-eye updates,
validation, current-checkpoint loading and exact next-update replay. Unknown or
changed evidence never authorizes skipping the full probe.
"""
import json
import math
from pathlib import Path
from radon_bridge.runtime.state import atomic_write_json, file_sha256


def read(path):
    return json.loads(Path(path).read_text())


def validate(path, spec_sha, hardware, *, full):
    path = Path(path)
    row = read(path)
    if (row.get('status') != 'accepted' or row.get('spec_sha256') != spec_sha or
            row.get('hardware') != hardware or row.get('test_used') is not False or
            row.get('formal_updates') != 0 or row.get('max_eyes') != 2 or
            row.get('full_state_replay_exact') is not True or
            row.get('fresh_train_batches') is not True or
            row.get('warmup_updates', 0) < 5 or row.get('measured_updates', 0) < 20 or
            row.get('step_memory_observed') is not True):
        raise ValueError('Resource evidence identity or acceptance differs')
    if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] <= 0
           for k in ('peak_gpu_gib', 'peak_step_memory_gib')):
        raise ValueError('Unknown measured resource peak')
    if full and (row.get('full_development') is not True or row.get('full_train_read') is not True):
        raise ValueError('Whole-cohort resource evidence absent')
    required = {'boundary.pt', 'expected.pt', 'resumed.pt'}
    if not required.issubset(row.get('files', {})):
        raise ValueError('Incomplete resource recovery files')
    for name in required:
        if file_sha256(path.parent/name) != row['files'][name]:
            raise ValueError('Resource evidence file changed')
    return row


def prior(reference, spec_sha, hardware):
    if not reference:
        return None
    if file_sha256(reference['path']) != reference['sha256']:
        raise ValueError('Closed profile receipt changed')
    validate(reference['path'], spec_sha, hardware, full=True)
    return dict(reference)


def qualify(reference, current, spec_sha, hardware, checkpoint_sha, *,
            total_gpu_gib, other_gpu_gib, allocated_ram_gib, other_ram_gib,
            allocated_cpus, worker_cpus, worker_ram_gib, output):
    """Join static full-cohort coverage with newly measured dynamic recovery.

    Envelope values are actual allocation/device observations, not requested
    resources. The worker step cap and allocation-wide 15% headroom are
    independent constraints. Account conservatively for all concurrent work.
    """
    old = validate(reference['path'], spec_sha, hardware, full=True)
    if file_sha256(reference['path']) != reference['sha256']:
        raise ValueError('Closed profile receipt changed')
    now = validate(current, spec_sha, hardware, full=False)
    if now.get('checkpoint_source_sha256') != checkpoint_sha:
        raise ValueError('Current checkpoint was not probed')
    if not all(type(x) in (int, float) and math.isfinite(x) for x in
               (total_gpu_gib, other_gpu_gib, allocated_ram_gib, other_ram_gib,
                allocated_cpus, worker_cpus, worker_ram_gib)):
        raise ValueError('Unknown resource envelope')
    if min(total_gpu_gib, allocated_ram_gib, allocated_cpus, worker_cpus, worker_ram_gib) <= 0 or min(other_gpu_gib, other_ram_gib) < 0:
        raise ValueError('Unknown resource envelope')
    if worker_cpus > allocated_cpus:
        raise ValueError('CPU allocation insufficient')
    gpu = max(old['peak_gpu_gib'], now['peak_gpu_gib'])
    ram = max(old['peak_step_memory_gib'], now['peak_step_memory_gib'])
    if other_gpu_gib + gpu*1.2 + 2 > min(.875*total_gpu_gib, total_gpu_gib-10):
        raise ValueError('GPU reserve insufficient')
    if worker_ram_gib > allocated_ram_gib:
        raise ValueError('Worker memory exceeds allocation')
    if ram > worker_ram_gib:
        raise ValueError('Worker step memory limit insufficient')
    if other_ram_gib + ram > .85*allocated_ram_gib:
        raise ValueError('Host memory reserve insufficient')
    record = dict(schema='radon_native_requalification_v1', status='accepted',
        spec_sha256=spec_sha, hardware=hardware, checkpoint_sha256=checkpoint_sha,
        full_cohort=dict(reference), current=dict(path=str(current), sha256=file_sha256(current)),
        full_state_replay_exact=True, static_coverage_reused=True,
        peak_gpu_gib=gpu, peak_step_memory_gib=ram, test_access=False,
        limits=dict(total_gpu_gib=total_gpu_gib, other_gpu_gib=other_gpu_gib,
            allocated_ram_gib=allocated_ram_gib, other_ram_gib=other_ram_gib,
            allocated_cpus=allocated_cpus, worker_cpus=worker_cpus,
            worker_ram_gib=worker_ram_gib))
    atomic_write_json(record, Path(output))
    return record


def live_envelope(torch, job, step):
    """Fail closed unless other steps are only small allocation-control owners.

    Unknown companion GPU peaks cannot be inferred from one nvidia-smi sample.
    Scientific companions therefore require their own future shared admission
    contract; this optimization does not admit them implicitly.
    """
    import subprocess
    def fields(line):
        return dict(x.split('=',1) for x in line.split() if '=' in x)
    jobrow=fields(subprocess.check_output(['scontrol','show','job',str(job),'-o'],text=True,timeout=20))
    tres=fields(jobrow.get('AllocTRES','').replace(',', ' '))
    def gib(text):
        units={'K':1/1024**2,'M':1/1024,'G':1,'T':1024}
        if not text or text[-1] not in units:raise ValueError('Unknown Slurm memory unit')
        value = float(text[:-1])*units[text[-1]]
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Unknown Slurm memory limit')
        return value
    ram=gib(tres.get('mem')); cpus=int(tres['cpu']);other_ram=0;other_cpus=0;worker_cpus=None;worker_ram=None
    if cpus <= 0:raise ValueError('Unknown Slurm CPU limit')
    lines=subprocess.check_output(['scontrol','show','step',str(job),'-o'],text=True,timeout=20)
    for line in lines.splitlines():
        row=fields(line)
        if row.get('State')!='RUNNING' or row.get('StepId','').endswith(('.extern','.batch')):continue
        resource=fields(row.get('TRES','').replace(',', ' '))
        m=gib(resource.get('mem'));c=int(row['CPUs'])
        if c <= 0:raise ValueError('Unknown Slurm CPU limit')
        if row.get('StepId')==str(job)+'.'+str(step):
            if worker_cpus is not None:raise ValueError('Duplicate worker step')
            worker_cpus=c;worker_ram=m;continue
        if m>2 or c>1:raise ValueError('Unmeasured concurrent scientific worker forbids profile reuse')
        other_ram+=m;other_cpus+=c
    if worker_cpus is None or worker_cpus+other_cpus>cpus:raise ValueError('Actual CPU step envelope unknown')
    if worker_ram is None or worker_ram > ram or other_ram >= ram:
        raise ValueError('Actual worker memory envelope unknown')
    free,total=torch.cuda.mem_get_info(0)
    return dict(total_gpu_gib=total/1024**3,other_gpu_gib=(total-free)/1024**3,
        allocated_ram_gib=ram,other_ram_gib=other_ram,allocated_cpus=cpus-other_cpus,worker_cpus=worker_cpus,worker_ram_gib=worker_ram)


def hardware_identity(torch):
    import subprocess
    prop=torch.cuda.get_device_properties(0)
    return dict(name=prop.name,total_bytes=prop.total_memory,torch=torch.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip())
