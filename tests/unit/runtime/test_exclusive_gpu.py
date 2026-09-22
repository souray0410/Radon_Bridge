import fcntl
import os
from types import SimpleNamespace
import pytest
import psutil
from radon_bridge.runtime import exclusive_gpu


def setup(monkeypatch,tmp_path,pids=''):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-fixture')
    monkeypatch.setenv('RESEARCH_GPU_LOCK_ROOT',str(tmp_path))
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=85,total=100))
    def query(cmd,**kwargs):
        if '--query-gpu=uuid' in cmd:return 'GPU-fixture\n'
        if '--query-compute-apps=pid' in cmd:return pids
        if '--query-gpu=memory.free' in cmd:return '4096\n'
        raise AssertionError(cmd)
    monkeypatch.setattr(exclusive_gpu.subprocess,'check_output',query)


def test_no_fixed_ten_gib_reserve_and_existing_lock_respected(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    handle=exclusive_gpu.acquire()
    try:
        with pytest.raises(BlockingIOError):exclusive_gpu.acquire()
    finally:handle.close()
    exclusive_gpu.acquire().close()


def test_unmanaged_compute_owner_rejected_and_lock_released(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path,pids=str(os.getpid()+1000))
    with pytest.raises(MemoryError,match='other GPU compute'):exclusive_gpu.acquire()
    with (tmp_path/'GPU-fixture.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)


def test_host_headroom_remains_required(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=14,total=100))
    with pytest.raises(MemoryError,match='Host reserve'):exclusive_gpu.acquire()
