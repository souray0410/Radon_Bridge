import copy
import torch
from torch.utils.data import Dataset
from tests.unit.models.test_native_group import fixture
from radon_bridge.data.observed_group import collate_group
from radon_bridge.models.native_group import NativeGroup
from radon_bridge.training.paired_native import train,DEFAULTS
from radon_bridge.runtime.host_checkpoint import capture_rng,restore_rng
from radon_bridge.studies.project_profile import same,profile_model
from radon_bridge.analysis.group_diagnostics import summarize_model


class SmallData(Dataset):
    collate_fn=staticmethod(collate_group)
    def __init__(self,batch,split,n=32):
        self.split=split;self.augment=False;self.epoch=0
        self.participant_ids=[split+str(i) for i in range(n)]
        self.counts=[1]*n
        self.rows=[dict(inputs={k:x[i%3:i%3+1].clone() for k,x in batch['inputs'].items()},
            labels={k:int(y[i%2]) for k,y in batch['labels'].items()},participant_id=self.participant_ids[i]) for i in range(n)]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]
    def set_epoch(self,e):self.epoch=e


def test_group_real_training_loop_resume_and_read_only_diagnosis(tmp_path):
    torch.set_num_threads(2)
    parents,shapes,sources,batch=fixture();cfg=copy.deepcopy(DEFAULTS)
    names=[s['key']+'_stage3' for s in sources]
    bridges=[dict(nodes=names,family='mmtm',hidden_dimension=8)]
    build=lambda:NativeGroup(parents,shapes,sources,bridges)
    original=build();train_data=SmallData(batch,'train');dev=SmallData(batch,'development',4)
    initial=copy.deepcopy(original.state_dict());rng=capture_rng()
    uninterrupted=tmp_path/'uninterrupted';resumed=tmp_path/'resumed'
    assert train(original,train_data,dev,cfg,3416,uninterrupted,'fixture',torch.device('cpu'),preflight_updates=2)['state']=='paused'
    second=build();second.load_state_dict(initial);restore_rng(rng)
    assert train(second,train_data,dev,cfg,3416,resumed,'fixture',torch.device('cpu'),preflight_updates=1)['state']=='paused'
    second=build()
    assert train(second,train_data,dev,cfg,3416,resumed,'fixture',torch.device('cpu'),preflight_updates=1)['state']=='paused'
    a=torch.load(uninterrupted/'last.pt',weights_only=False);b=torch.load(resumed/'last.pt',weights_only=False)
    for key in ('model','optimizer','scheduler','rng','node_ids'):assert same(a[key],b[key]),key
    assert a['progress']['offset']==b['progress']['offset']
    assert not (resumed/'accepted.json').exists()
    before=copy.deepcopy(second.state_dict());second.train();rng=capture_rng()
    summarize_model(second,SmallData(batch,'train',4),tmp_path/'diagnostic',torch.device('cpu'),lambda:False)
    assert second.training and same(before,second.state_dict()) and same(rng,capture_rng())
    resource=profile_model(second,collate_group([train_data[0]]),cfg,'fixture',tmp_path/'profile',torch.device('cpu'),production=False)
    assert resource['exact_next_update'] and not resource['production_gpu_admission']
    assert same(before,second.state_dict())
