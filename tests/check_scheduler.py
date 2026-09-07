"""Exercise scheduling, accounting, overwrite protection and group budget gates without GPUs."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('controller','scripts/run_integer_experiment.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
processes=[]
class Process:
    def __init__(self,cmd,**kw):
        self.pid=1000+len(processes);self.gpu=int(kw['env']['CUDA_VISIBLE_DEVICES']);processes.append(self);self.returncode=None;self.ticks=0
        directory=Path(cmd[cmd.index('--output')+1]);m.write_json(directory/'summary.json',{'passed':True})
    def poll(self):
        self.ticks+=1
        if self.ticks>=2:self.returncode=0
        return self.returncode
    def terminate(self):self.returncode=-15
    def kill(self):self.returncode=-9
    def wait(self,**kw):self.returncode=0;return 0

with tempfile.TemporaryDirectory() as tmp:
    c=m.Controller.__new__(m.Controller);c.root=Path(tmp);c.args=SimpleNamespace(phase='preflight',data='/unused')
    c.gpus=[0,1];c.protocol={'prior_gpu_minutes':13.341};c.limit=60;c.ledger={'jobs':[]};c.active={};c.peak={}
    c.phase='test';c.stop=False;c.stop_reason=None;c.commit=None
    with patch.object(m,'devices',return_value={0:{'free':20000,'uuid':'GPU-0'},1:{'free':20000,'uuid':'GPU-1'}}), \
         patch.object(m.subprocess,'Popen',Process), \
         patch.object(m.subprocess,'check_output',side_effect=lambda *a,**k:'\n'.join(f'{p.pid}, GPU-{p.gpu}, 5000' for p in processes if p.returncode is None)), \
         patch.object(m.time,'sleep',lambda _:None):
        jobs=[m.job(str(i),m.config(1,3e-5)) for i in range(2)]
        result=c.run_jobs(jobs)
        assert len(result)==2 and {j['gpu'] for j in c.ledger['jobs']}=={0,1}
        assert not c.active and all(p.returncode==0 for p in processes)
        assert abs(c.used()-sum(j['gpu_seconds'] for j in c.ledger['jobs'])/60)<1e-12
        assert c.group_fits([m.job('next',m.config(1,3e-5,epochs=6))],10)
        c.limit=c.used()+.01
        assert not c.group_fits([m.job('too_big',m.config(1,3e-5,epochs=6))],10)
        try:c.start(jobs[0],0)
        except RuntimeError:pass
        else:raise AssertionError('Overwrite allowed')
        c.limit=60;c.gpus=[1]
        single=c.run_jobs([m.job('single2',m.config(1,3e-5)),m.job('single3',m.config(1,3e-5))])
        assert len(single)==2 and all(j['gpu']==1 for j in c.ledger['jobs'][-2:])
    print(json.dumps({'two_gpu_dispatch':True,'summed_gpu_accounting':True,'comparison_group_budget_gate':True,'overwrite_rejected':True,'single_gpu_allowlist':True}))

# Explicitly unlimited studies must still dispatch and account beyond the old cap.
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp); protocol=root/'protocol.json'
    m.write_json(protocol,{'review_status':'approved','schema':'two_stage_ratio_convergence_v3',
        'prior_gpu_minutes':500,'max_gpu_minutes':None,'gpu_time_policy':'unlimited_until_convergence'})
    args=SimpleNamespace(output=str(root),protocol=str(protocol),phase='preflight',data='/unused')
    c=m.Controller(args)
    assert c.group_fits([m.job('large',m.config(1,3e-5,epochs=60))],10000)
    state=json.loads((root/'status.json').read_text())
    assert state['budget_gpu_minutes'] is None and state['cumulative_gpu_minutes']==500
    assert 'Infinity' not in (root/'status.json').read_text()
    c.shutdown()
    __import__("atexit").unregister(c.shutdown)
    print(json.dumps({'unlimited_accepts_prior_over_240':True,'unlimited_group_gate':True,'valid_json_null_limit':True}))
