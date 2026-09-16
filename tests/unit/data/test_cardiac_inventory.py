import json
from pathlib import Path
import pytest
from radon_bridge.data.cardiac_inventory import audit
from radon_bridge.runtime.state import file_sha256


def write(path,record):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record));return path


def fixture(tmp_path):
    transfer=dict(state='all_selected_bytes_accepted',test_access=False,total_files=1,total_bytes=10,
        groups=[dict(group='field',files=1,bytes=10,manifest_sha256='a'*64)])
    write(tmp_path/'multiorgan/manifests/transfer_completed_20260916.json',transfer)
    receipt=write(tmp_path/'ophthalmology/accepted.json',dict(state='accepted'))
    data=tmp_path/'ophthalmology/data.csv';data.write_bytes(b'eid,f.31.0.0,p53_i0\n\xff\xfe participant rows must never decode\n')
    refs=dict(receipt_path='ophthalmology/accepted.json',receipt_sha256=file_sha256(receipt),
        files=[dict(path='ophthalmology/data.csv',bytes=data.stat().st_size,sha256=file_sha256(data))])
    write(tmp_path/'multiorgan/source_metadata/shared_phenotype_references.json',refs)
    return data,receipt


def test_header_only_never_accepts_semantics_or_reads_participant_rows(tmp_path):
    fixture(tmp_path);r=audit(tmp_path)
    assert r['files'][0]['columns']==3
    assert r['files'][0]['unique_field_ids']==[31,53]
    assert r['participant_rows_read']==0
    assert not r['semantic_data_acceptance'] and not r['project_training_authorized']


def test_changed_reference_size_rejected(tmp_path):
    data,_=fixture(tmp_path);data.write_bytes(b'changed')
    with pytest.raises(ValueError,match='size changed'):audit(tmp_path)


def test_changed_acceptance_receipt_rejected(tmp_path):
    _,receipt=fixture(tmp_path);receipt.write_text('{}')
    with pytest.raises(ValueError,match='receipt changed'):audit(tmp_path)
