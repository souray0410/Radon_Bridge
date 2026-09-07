"""Deterministic scheduler regression: refill, backpressure, failure, drain, memory."""
import io,json,math,tempfile,threading,time as real_time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.rolling_dispatch import RollingController

class Clock:
    def __init__(self):self.now=0
    def monotonic(self):return self.now
    def sleep(self,_):self.now+=.25;real_time.sleep(.001)

class Process:
    def __init__(self,clock,pid,duration,code=0):self.clock=clock;self.pid=pid;self.end=clock.now+duration;self.code=code;self.killed=False
    def poll(self):return -15 if self.killed else self.code if self.clock.now>=self.end else None
    def terminate(self):self.killed=True
    def kill(self):self.killed=True

class Fixture(RollingController):
    def start(self,job,gpu):
        self.before_start(job,gpu,self.root/job['id'])
        key=job['id'];path=self.root/key;path.mkdir()
        summary=dict(state='complete',converged_by_policy=key!=self.nonplateau,stop_reason='validation_plateau',test_used=False)
        (path/'summary.json').write_text(json.dumps(summary))
        p=Process(self.clock,100+len(self.launched),job['duration'])
        self.active[key]=dict(process=p,gpu=gpu,start=self.clock.now,stop_at=None,log=io.StringIO())
        self.launched.append((key,gpu,self.clock.now));self.processes[key]=p
        if key=='C':self.refill_observed.set()


def scenario(kind):
    with tempfile.TemporaryDirectory() as tmp:
        clock=Clock();c=Fixture.__new__(Fixture);c.clock=clock;c.root=Path(tmp)
        c.args=SimpleNamespace(phase='run');c.phase='run';c.commit='fixture';c.protocol=dict(prior_gpu_minutes=0,min_free_gpu_mib=12288)
        c.ledger={'jobs':[]};c.active={};c.peak={};c.stop=False;c.stop_reason=None;c.limit=math.inf;c.gpus=[1,0]
        c.archivals={};c.max_archivals=4;c.pending_count=0;c.on_status=None;c.pause_file=c.root/'drain.request'
        c.launched=[];c.published=[];c.processes={};c.refill_observed=threading.Event();c.nonplateau='A' if kind=='nonplateau' else None
        c.used=lambda:0
        def before(job,gpu,path):
            if kind=='storage' and job['id']=='C':raise RuntimeError('storage_needs_attention')
        def finish(job,path):
            if kind=='refill' and job['id']=='A':
                assert c.refill_observed.wait(3),'Archive waiting must not block free-GPU refill'
            if kind=='archive_failure' and job['id']=='A':raise RuntimeError('hash_mismatch')
            return job['id']
        c.before_start=before;c.finish_background=finish;c.publish_result=lambda job,value:c.published.append(value)
        def gpu_memory(*_,**__):
            if kind=='drain' and clock.now>=.5:c.pause_file.touch()
            if kind=='memory' and clock.now>=.5:return '100, GPU-1, 10241\n99999, GPU-0, 25000\n'
            return '99999, GPU-0, 25000\n' # Unrelated workload must never be terminated.
        jobs=[dict(id='A',duration=1,config={'seed':1}),dict(id='B',duration=6,config={'seed':2}),dict(id='C',duration=1,config={'seed':3}),dict(id='D',duration=1,config={'seed':4})]
        caught=None
        with patch('scripts.rolling_dispatch.time',clock),patch('scripts.rolling_dispatch.devices',return_value={0:{'free':16000,'uuid':'GPU-0'},1:{'free':16000,'uuid':'GPU-1'}}),patch('scripts.rolling_dispatch.subprocess.check_output',side_effect=gpu_memory):
            try:c.run_jobs(jobs)
            except (RuntimeError,InterruptedError) as exc:caught=exc
        if kind=='refill':
            assert caught is None,caught
            launch={k:(g,t) for k,g,t in c.launched}
            assert launch['C'][0]==launch['A'][0] and launch['C'][1]<c.processes['B'].end
            assert sorted(c.published)==['A','B','C','D']
            assert len(c.ledger['jobs'])==4 and len({x['id'] for x in c.ledger['jobs']})==4
        elif kind in ('nonplateau','storage'):
            assert isinstance(caught,RuntimeError) and [x[0] for x in c.launched]==['A','B']
            assert not c.processes['B'].killed and 'B' in c.published
        elif kind=='archive_failure':assert isinstance(caught,RuntimeError) and 'hash_mismatch' in str(caught)
        elif kind=='drain':
            assert isinstance(caught,InterruptedError) and [x[0] for x in c.launched]==['A','B']
            assert sorted(c.published)==['A','B'] and not any(p.killed for p in c.processes.values())
        elif kind=='memory':assert isinstance(caught,InterruptedError) and c.stop_reason=='memory_limit'
        return dict(scenario=kind,passed=True,launched=[x[0] for x in c.launched])

if __name__=='__main__':print(json.dumps(dict(passed=True,checks=[scenario(k) for k in ('refill','nonplateau','storage','archive_failure','drain','memory')]),indent=2))
