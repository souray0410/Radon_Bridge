import hashlib
import json
from pathlib import Path

import pytest
from radon_bridge.studies.import_selected_parent import publish
from radon_bridge.models.native_materialization import verify_selected
from radon_bridge.runtime.state import file_sha256


def fixture(tmp_path):
    root = tmp_path / 'source'; root.mkdir()
    spec = dict(framework=dict(api='V5'), test_used=False)
    (root/'spec.json').write_text(json.dumps(spec))
    for name in ('best.pt', 'history.json', 'development_predictions.npz'):
        (root/name).write_bytes(b'opaque immutable test content')
    files = {p.name:file_sha256(p) for p in root.iterdir()}
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    (root/'source_accepted.json').write_text(json.dumps(dict(identity=identity, test_used=False, status='accepted', files=files.copy())))
    files['source_accepted.json'] = file_sha256(root/'source_accepted.json')
    manifest = dict(schema='look_selected_native_v2',test_access=False,complete_training_resume=False,
                    complete_task_model=True,selected_prediction_replay=True,files=files)
    (root/'selected_artifact.json').write_text(json.dumps(manifest))
    return root, file_sha256(root/'selected_artifact.json')


def test_preserves_content_provenance_and_idempotency(tmp_path):
    src, digest = fixture(tmp_path); calls=[]
    def check(path):
        verify_selected(path); calls.append(path)
    target = publish(src,tmp_path/'out',digest,check)
    accepted,_,_ = verify_selected(target)
    assert accepted['schema']=='radon_bridge_selected_native_v2'
    assert file_sha256(target/'best.pt')==file_sha256(src/'best.pt')
    assert file_sha256(target/'provenance/imported_selected_manifest.json')==digest
    assert publish(src,tmp_path/'out',digest,check)==target
    assert len(calls)==2
    assert not list((tmp_path/'out').glob('*.partial'))


@pytest.mark.parametrize('mutation', ['weight', 'manifest', 'v4', 'test', 'escape'])
def test_reject_changed_unmigrated_or_unsafe_sources(tmp_path, mutation):
    src,digest=fixture(tmp_path)
    if mutation=='weight': (src/'best.pt').write_bytes(b'changed')
    elif mutation=='manifest': digest='0'*64
    else:
        m=json.loads((src/'selected_artifact.json').read_text())
        if mutation=='escape':
            outside=tmp_path/'outside'; outside.write_bytes(b'outside')
            m['files']['../outside']=file_sha256(outside)
        else:
            spec=json.loads((src/'spec.json').read_text())
            if mutation=='v4':spec['framework']['api']='V4'
            else:spec['test_used']=True
            (src/'spec.json').write_text(json.dumps(spec));m['files']['spec.json']=file_sha256(src/'spec.json')
        (src/'selected_artifact.json').write_text(json.dumps(m));digest=file_sha256(src/'selected_artifact.json')
    with pytest.raises(ValueError): publish(src,tmp_path/'out',digest,lambda p:None)
    assert not (tmp_path/'out').exists()


def test_failed_consumer_cannot_publish(tmp_path):
    src,digest=fixture(tmp_path)
    def fail(path):raise ValueError('strict real model load failed')
    with pytest.raises(ValueError,match='real model'):publish(src,tmp_path/'out',digest,fail)
    assert len(list((tmp_path/'out').glob('*.partial')))==1
    assert not [p for p in (tmp_path/'out').iterdir() if p.is_dir() and not p.name.endswith('.partial')]
