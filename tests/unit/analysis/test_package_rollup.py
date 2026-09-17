import csv
import json
from pathlib import Path
import pytest
from radon_bridge.analysis.package_rollup import publish
from radon_bridge.runtime.state import file_sha256


def setup(tmp_path):
    spec=tmp_path/'spec.json';spec.write_text(json.dumps({'model':{'name':'resnet50'},'disease':'glaucoma','seed':3416}))
    return [dict(spec=str(spec),spec_sha256=file_sha256(spec),run_dir=str(tmp_path/'case'))]


def accept(tmp_path):
    package=tmp_path/'case/packages/core';(package/'report').mkdir(parents=True)
    (package/'accepted.json').write_text('{}')
    with (package/'report/metrics.csv').open('w') as stream:
        stream.write('arm,cfp_f1,oct_f1,mean_f1\nsvd_radon,0.6,0.5,0.55\n')


def test_pending_not_zero_idempotent_and_incremental(tmp_path):
    tasks=setup(tmp_path);out=tmp_path/'publish'
    a=publish(tasks,out,verify=lambda *a:None,renderer=lambda *a:None)
    assert a['accepted']==0
    assert publish(tasks,out,verify=lambda *a:None,renderer=lambda *a:None)['state']=='unchanged'
    accept(tmp_path)
    b=publish(tasks,out,verify=lambda *a:None,renderer=lambda *a:None)
    assert b['accepted']==1 and b['snapshot']!=a['snapshot']
    with (out/'current/results.csv').open() as stream:assert len(list(csv.DictReader(stream)))==1


def test_failure_preserves_last_good_current_and_tampering_rejected(tmp_path):
    tasks=setup(tmp_path);out=tmp_path/'publish'
    publish(tasks,out,verify=lambda *a:None,renderer=lambda *a:None);old=(out/'current').resolve()
    accept(tmp_path)
    def fail(*a):raise RuntimeError('plot failure')
    with pytest.raises(RuntimeError):publish(tasks,out,verify=lambda *a:None,renderer=fail)
    assert (out/'current').resolve()==old
    with pytest.raises(ValueError,match='Duplicate'):publish(tasks+tasks,out,verify=lambda *a:None)
