"""Refill each GPU independently; accept/archive results without blocking dispatch.

The existing worker command, CUDA assignment, allocator, optimizer, RNG and
training protocol are unchanged. Only independent job launch timing changes.
"""
from concurrent.futures import ThreadPoolExecutor
import subprocess,time
from scripts.run_integer_experiment import Controller,devices,read_json
from radonbridge.experiment import write_json

class RollingController(Controller):
    def __init__(self,args,*,before_start,finish_background,publish_result,on_status=None,pause_file=None,max_archivals=4):
        self.before_start=before_start;self.finish_background=finish_background
        self.publish_result=publish_result;self.on_status=on_status;self.pause_file=pause_file
        self.max_archivals=max_archivals;self.archivals={};self.pending_count=0
        super().__init__(args)

    def status(self,state,**kw):
        super().status(state,**kw)
        if self.on_status:
            self.on_status(state,active=[dict(id=k,pid=v['process'].pid,gpu=v['gpu']) for k,v in self.active.items()],
                           queued_count=self.pending_count,archivals_pending=len(self.archivals),**getattr(self,'allocation_status',{}),**kw)

    def start(self,job,gpu):
        self.before_start(job,gpu,self.root/job['id'])
        super().start(job,gpu)

    def run_jobs(self,jobs,allow_oom=False):
        if allow_oom:raise ValueError('Rolling performance jobs cannot accept OOM as completion')
        pending=list(jobs);lookup={j['id']:j for j in jobs}
        if len(lookup)!=len(pending):raise ValueError('Duplicate job IDs')
        results={};error=None;drain=False
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='rb_evidence') as pool:
            while pending or self.active or self.archivals:
                from scripts.gpu_allocation import refresh_controller
                info=devices();allowed=refresh_controller(self,info)
                # Only the main thread mutates result indices or retires work copies.
                for future,key in list(self.archivals.items()):
                    if not future.done():continue
                    del self.archivals[future]
                    try:self.publish_result(lookup[key],future.result())
                    except Exception as exc:error=error or exc
                if self.pause_file and self.pause_file.exists():drain=True
                if error is not None or drain:pending.clear()
                if self.stop:
                    pending.clear()
                    for v in self.active.values():
                        if v['stop_at'] is None:v['process'].terminate();v['stop_at']=time.monotonic()
                        elif time.monotonic()-v['stop_at']>20:v['process'].kill()
                # Check own process memory before launching another independent job.
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
                from scripts.gpu_allocation import memory_snapshot
                memory,per_gpu,project_on=memory_snapshot(raw,self.active,info)
                for key,v in self.active.items():
                    used=memory.get(v['process'].pid,0);self.peak[key]=max(self.peak.get(key,0),used)
                if hasattr(self,'allocation_status'):self.allocation_status['project_memory_mib_by_gpu']=per_gpu
                if any(x>10240 for x in per_gpu.values()):self.stop=True;self.stop_reason='memory_limit'
                if self.used()+(20*len(self.active)+2)/60>=self.limit:self.stop=True;self.stop_reason='budget'
                for key,v in list(self.active.items()):
                    code=v['process'].poll()
                    if code is None:continue
                    v['log'].close();del self.active[key]
                    self.ledger['jobs'].append(dict(id=key,gpu=v['gpu'],gpu_seconds=time.monotonic()-v['start'],exit_code=code,
                                                   sampled_peak_process_mib=self.peak.get(key,0)))
                    write_json(self.root/'ledger.json',self.ledger)
                    try:
                        if code!=0:raise RuntimeError(f'Worker {key} exited {code}; inspect preserved attempt')
                        summary=read_json(self.root/key/'summary.json')
                        if not (summary.get('state')=='complete' and summary.get('converged_by_policy') and summary.get('stop_reason')=='validation_plateau' and summary.get('test_used') is False):
                            raise RuntimeError(f'Worker {key} did not satisfy completion policy')
                        results[key]=summary
                        self.archivals[pool.submit(self.finish_background,lookup[key],self.root/key)]=key
                    except Exception as exc:
                        if not self.stop:error=error or exc
                if error is not None or drain:pending.clear()
                # A completed worker frees its GPU immediately, even while its peer
                # still trains and evidence copy/hash checks run on the CPU.
                if pending and not self.stop and error is None and not drain and len(self.archivals)<self.max_archivals:
                    occupied={v['gpu'] for v in self.active.values()}
                    capacity=max(self.max_archivals,len(allowed))
                    for gpu in allowed:
                        if project_on.get(gpu):continue
                        if pending and len(self.archivals)+len(self.active)<capacity and gpu not in occupied and gpu in info and info[gpu]['free']>=self.protocol.get('min_free_gpu_mib',12288):
                            index=next((i for i,j in enumerate(pending) if j.get('gpu',gpu)==gpu),None)
                            if index is None:continue
                            try:self.start(pending[index],gpu);pending.pop(index)
                            except Exception as exc:error=exc;pending.clear();break
                self.pending_count=len(pending)
                state='stopping' if self.stop else 'draining_failure' if error is not None else 'draining' if drain else 'paused_dispatch' if pending and not allowed else 'running'
                self.status(state,error=repr(error) if error is not None else None)
                if pending or self.active or self.archivals:time.sleep(1)
        if error is not None:raise error
        if self.stop:raise InterruptedError(self.stop_reason)
        if drain:raise InterruptedError('drain_requested')
        return results
