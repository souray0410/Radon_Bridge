"""Arbitrary-card live selection, draining, pause/resume, and project-only memory."""
import copy,json,math,tempfile,threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.gpu_allocation import Allocation,publish,validate,memory_snapshot
from check_rolling_dispatch import Fixture,Clock

inventory={i:dict(uuid=f'GPU-{i}',free=24000) for i in range(6)}
with tempfile.TemporaryDirectory() as t:
    root=Path(t);path=root/'allocation.json';a=Allocation(path,[0,1],required=True)
    assert a.refresh(inventory)[0]['allocation_error']
    for indices in ([2],[0,5],list(range(6)),[]):
        publish(path,indices,inventory);v,changed=a.refresh(inventory)
        assert v['selected_gpu_indices']==indices and changed
        assert not a.refresh(inventory)[1]
    for bad in ([True],[-1],[0,0],[9],'0'):
        try:validate(bad,inventory)
        except ValueError:pass
        else:raise AssertionError(bad)
    path.write_text('{broken');assert a.refresh(inventory)[0]['selected_gpu_indices']==[]
    path.unlink();assert a.refresh(inventory)[0]['allocation_error']
    assert Allocation(path,[0,1],required=True).refresh(inventory)[0]['allocation_error']
    # Another project's large usage must not enter R&B's 10 GiB/card total.
    active={'ours':dict(process=SimpleNamespace(pid=101),gpu=0)}
    bypid,usage,on=memory_snapshot('101, GPU-0, 7000\n202, GPU-0, 18000\n303, GPU-0, 1000\n',active,inventory,identify=lambda p:p==303)
    assert usage=={0:8000} and on=={0:[101,303]} and bypid[202]==18000
    assert memory_snapshot('101, GPU-0, 9500\n303, GPU-0, 1000\n',active,inventory,identify=lambda p:p==303)[1][0]>10240
    clock=Clock();c=Fixture.__new__(Fixture);c.clock=clock;c.root=root
    c.args=SimpleNamespace(phase='run');c.phase='run';c.commit='fixture';c.protocol=dict(prior_gpu_minutes=0,min_free_gpu_mib=12288)
    c.ledger={'jobs':[]};c.active={};c.peak={};c.stop=False;c.stop_reason=None;c.limit=math.inf;c.gpus=[0,2,5]
    c.allocation=Allocation(path,c.gpus,required=True);c.allocation_status={}
    c.archivals={};c.max_archivals=4;c.pending_count=0;c.pause_file=root/'drain.request'
    c.launched=[];c.published=[];c.processes={};c.refill_observed=threading.Event();c.nonplateau=None;c.used=lambda:0
    states=[];peak=[0];chosen={};phase=[None]
    c.on_status=lambda state,**kw:(states.append(state),peak.__setitem__(0,max(peak[0],len(c.active))))
    def before(job,gpu,unused):
        chosen[job['id']]=list(c.allocation_status['selected_gpu_indices']);assert gpu in chosen[job['id']]
    c.before_start=before;c.finish_background=lambda job,path:job['id'];c.publish_result=lambda job,value:c.published.append(value)
    def devices():
        now=clock.now
        selected=[0,2,5] if now<.25 else [2] if now<2 else [] if now<3 else [0,2] if now<4 else list(range(6))
        if selected!=phase[0]:publish(path,selected,inventory);phase[0]=selected
        return inventory
    jobs=[dict(id=f'job{i}',duration=6 if i==0 else 4,config=dict(seed=i)) for i in range(20)];original=copy.deepcopy(jobs)
    with patch('scripts.rolling_dispatch.time',clock),patch('scripts.rolling_dispatch.devices',side_effect=devices),patch('scripts.rolling_dispatch.subprocess.check_output',return_value='99999, GPU-0, 18000\n'):
        c.run_jobs(jobs)
    assert len(c.published)==len(set(c.published))==20 and jobs==original
    assert peak[0]>=5 and 'paused_dispatch' in states
    assert not any(p.killed for p in c.processes.values())
    assert len(c.ledger['allocation_events'])==5,c.ledger['allocation_events']
    assert all(not (2<=t<3) for _,_,t in c.launched)
    assert any(g not in (0,1) for _,g,_ in c.launched)
print(json.dumps(dict(passed=True,arbitrary_six_devices=True,single_dual_multi_pause_resume=True,active_jobs_not_killed=True,
    completed_exactly_once=True,unrelated_memory_excluded=True,all_project_workers_included=True,invalid_selection_pauses=True)))
