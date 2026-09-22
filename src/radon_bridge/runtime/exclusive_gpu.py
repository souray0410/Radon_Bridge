"""Exclusive, finite workstation qualification using the existing device lock."""
import fcntl
import os
from pathlib import Path
import subprocess


def acquire():
    import psutil
    memory = psutil.virtual_memory()
    if memory.available < .15 * memory.total:
        raise MemoryError('Host reserve below 15 percent')
    device = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not device or ',' in device:
        raise ValueError('One explicit CUDA device required')
    def query(option):
        return subprocess.check_output(['nvidia-smi', '-i', device, option,
            '--format=csv,noheader,nounits'], text=True).strip()
    uuid = query('--query-gpu=uuid')
    if not uuid.startswith('GPU-') or '/' in uuid or '\n' in uuid:
        raise ValueError('Expected one physical GPU UUID')
    root = Path(os.environ['RESEARCH_GPU_LOCK_ROOT'])
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / (uuid + '.lock')).open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Non-cooperating users may not hold our lock. Never probe on top of
        # their compute processes; shared admission belongs to the scheduler.
        pids = query('--query-compute-apps=pid').splitlines()
        if any(p.strip() != str(os.getpid()) for p in pids):
            raise MemoryError('Exclusive qualification requires no other GPU compute process')
        free = int(query('--query-gpu=memory.free'))
        if free <= 0:
            raise MemoryError('No currently available GPU memory')
        return handle
    except BaseException:
        handle.close()
        raise
