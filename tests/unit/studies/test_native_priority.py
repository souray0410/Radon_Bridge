import json
from pathlib import Path
import pytest
from radon_bridge.studies.native_prerequisites import collect
from radon_bridge.runtime.state import file_sha256

def test_screening_flag_preserved_and_other_ineligible_scope_excluded(tmp_path):
    tasks=[]
    for i,scope in enumerate(('expanded_cohort_native_screening','unapproved_scope')):
        spec=dict(model=dict(name='resnet50',spatial_dims=3),track='oct_volume_3d',
            disease='cataract',training=dict(seed=3416),test_used=False,
            eligible_for_formal_selection=False,scope=scope,
            train_manifest_sha256='train',development_manifest_sha256='dev',
            cache_receipt_sha256='cache',aggregation='valid_eye_feature_mean_v1')
        path=tmp_path/f'{i}.json';path.write_text(json.dumps(spec))
        tasks.append(dict(spec=str(path),spec_sha256=file_sha256(path),run_dir=str(tmp_path/f'run{i}')))
    queue=tmp_path/'queue.json';queue.write_text(json.dumps(dict(tasks=tasks)))
    catalog=collect([queue])
    assert len(catalog['candidates'])==1
    assert catalog['candidates'][0]['run_dir']==str(tmp_path/'run0')
    for task in tasks:
        assert file_sha256(Path(task['spec']))==task['spec_sha256']
        assert json.loads(Path(task['spec']).read_text())['eligible_for_formal_selection'] is False
    assert catalog['test_access'] is False
