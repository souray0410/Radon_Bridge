"""Bounded metadata readiness audit; never reads participant rows or picks labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

from radon_bridge.runtime.state import atomic_write_json, file_sha256, utc_now


def audit(root):
    root=Path(root).resolve()
    receipt=root/'multiorgan/manifests/transfer_completed_20260916.json'
    references=root/'multiorgan/source_metadata/shared_phenotype_references.json'
    transfer=json.loads(receipt.read_text());metadata=json.loads(references.read_text())
    if transfer.get('state')!='all_selected_bytes_accepted' or transfer.get('test_access') is not False:
        raise ValueError('Transfer acceptance missing or test provenance unclear')
    rows=transfer['groups']
    if (sum(r['files'] for r in rows)!=transfer['total_files'] or
        sum(r['bytes'] for r in rows)!=transfer['total_bytes']):
        raise ValueError('Transfer totals disagree')
    prior=(root/metadata['receipt_path']).resolve()
    if not prior.is_relative_to(root) or file_sha256(prior)!=metadata['receipt_sha256']:
        raise ValueError('Original phenotype receipt changed')
    files=[]
    for ref in metadata['files']:
        path=(root/ref['path']).resolve()
        if not path.is_relative_to(root) or path.stat().st_size!=ref['bytes']:
            raise ValueError('Referenced phenotype asset missing or size changed')
        row=dict(asset=path.name,bytes=ref['bytes'],original_sha256=ref['sha256'],current_size_verified=True)
        if path.suffix.lower()=='.csv':
            with path.open('rb') as stream:header=stream.readline(2*1024*1024)
            if not header.endswith(b'\n'):raise ValueError('CSV header exceeds bounded read')
            text=header.decode('utf-8-sig')
            dialect=csv.Sniffer().sniff(text,delimiters=',\t;')
            names=next(csv.reader([text],dialect))
            field_ids=set()
            for name in names:
                match=re.match(r'^(?:f\.|p)?(\d+)(?:[._-]|$)',name)
                if match:field_ids.add(int(match.group(1)))
            row.update(header_sha256=hashlib.sha256(header).hexdigest(),columns=len(names),
                       unique_field_ids=sorted(field_ids),participant_rows_read=0)
        files.append(row)
    return dict(schema='ukb_cardiac_metadata_audit_v1',updated_at=utc_now(),
        transfer_receipt_sha256=file_sha256(receipt),metadata_reference_sha256=file_sha256(references),
        files=files,groups=[{k:r[k] for k in ('group','files','bytes','manifest_sha256')} for r in rows],
        total_files=transfer['total_files'],total_bytes=transfer['total_bytes'],
        byte_evidence='prior independently accepted receipts plus current phenotype sizes; not a new full hash pass',
        participant_rows_read=0,test_performance_access=False,semantic_data_acceptance=False,
        pending=['field meaning and image/ECG decoding','participant/visit/eye pairing and acquisition time',
                 'clinical label source and diagnosis time','global split and parent exposure audit',
                 'counts and missingness by eligible task','user task protocol lock'],
        model_training_authorized=False,project_training_authorized=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();atomic_write_json(audit(args.root),Path(args.output))

if __name__=='__main__':main()
