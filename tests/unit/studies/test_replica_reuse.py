import json
import copy
import pytest
from radon_bridge.studies.autoresearch import reuse_replica
from radon_bridge.runtime.state import file_sha256


def fixture(tmp_path):
    spec={'test_used':False,'training':{'seed':3417,'lr':0.1},'framework':'pinned',
          'train_manifest_sha256':'train','recipe_selection':{'old':'nomination'}}
    p=tmp_path/'spec.json';p.write_text(json.dumps(spec))
    task={'id':'original','run_dir':str(tmp_path/'run'),'spec':str(p),'spec_sha256':file_sha256(p)}
    q=tmp_path/'queue.json';q.write_text(json.dumps({'test_used':False,'tasks':[task]}))
    return spec,task,[{'path':str(q),'sha256':file_sha256(q)}]


def test_selection_provenance_can_differ_but_execution_is_retained(tmp_path):
    expected,task,queues=fixture(tmp_path)
    expected['recipe_selection']={'new':'nomination'}
    assert reuse_replica(expected,queues)==task
    assert json.loads(open(task['spec']).read())['recipe_selection']=={'old':'nomination'}


@pytest.mark.parametrize('key,value',[('framework','different'),('training',{'seed':3418,'lr':0.1}),
    ('training',{'seed':3417,'lr':0.2}),('train_manifest_sha256','different')])
def test_scientific_differences_never_reuse(tmp_path,key,value):
    expected,task,queues=fixture(tmp_path);expected[key]=value
    assert reuse_replica(expected,queues) is None


def test_changed_reuse_queue_fails_closed(tmp_path):
    expected,task,queues=fixture(tmp_path);queues[0]['sha256']='bad'
    with pytest.raises(ValueError,match='queue changed'):reuse_replica(expected,queues)
