import json
import numpy as np
import pytest
from radon_bridge.analysis.cohort_report import report
from radon_bridge.evaluation.metrics import classification_metrics
from radon_bridge.studies.cohort_case import sha


def test_partial_publication_replay_and_tamper(tmp_path):
    y=np.arange(296)%2;p=np.stack([1-y,y],axis=1)*.8+.1
    case=tmp_path/'trials'/'none_seed3416';case.mkdir(parents=True)
    cfg={'seed':3416};spec=tmp_path/'spec.json';spec.write_text(json.dumps(cfg))
    np.savez(case/'selected_predictions.npz',ids=np.array([str(i) for i in range(296)]),y=y,cfp=p,oct=p)
    (case/'best.pt').write_bytes(b'test_model')
    files={name:sha(case/name) for name in ('selected_predictions.npz','best.pt')}
    receipt=dict(configuration=cfg,test_used=False,converged_by_policy=True,files=files,best_epoch=1,epochs_ran=8,
        selected_validation=dict(tasks={k:classification_metrics(y,p) for k in ('cfp','oct')}))
    (case/'accepted.json').write_text(json.dumps(receipt))
    q=dict(sequence_id='test',cases=[dict(id='none',name='none_seed3416',config=str(spec),provenance='accepted reference'),dict(id='svd',name='missing',config='absent',provenance='new')])
    (tmp_path/'queue.json').write_text(json.dumps(q))
    r=report(tmp_path);assert not r['complete'];assert r['results'][0]['mean_f1']==1
    dest=tmp_path/'publication/current.json';before=dest.read_bytes();stamp=dest.stat().st_mtime_ns
    report(tmp_path);assert dest.read_bytes()==before and dest.stat().st_mtime_ns==stamp
    (case/'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Evidence changed'):report(tmp_path)
    assert dest.read_bytes()==before
