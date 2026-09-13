import numpy as np
from radon_bridge.analysis.project_report import bootstrap
from radon_bridge.analysis.project_rollup import summarize


def test_shared_participant_resampling_and_zero_variance():
    y=np.array([0,0,1,1]);pred=np.stack([y,y])
    result=bootstrap(y,pred,[[1,-1]],100)
    assert result['participants']==4
    assert result['comparisons'][0]['zero_variance']
    assert result['comparisons'][0]['simultaneous_95'] is None


def test_empty_is_not_complete(tmp_path):
    assert summarize([],tmp_path)['complete'] is False
