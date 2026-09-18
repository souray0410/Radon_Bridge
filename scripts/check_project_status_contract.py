"""Mirror the shared PHD public project-status contract for local CI.

Keep this fail-closed and byte-for-byte compatible in semantics with
PHD/scripts/project_status.py validate(); it must never broaden accepted states.
"""
from datetime import datetime, timezone
import json
import re
from pathlib import Path

SCHEMA='research_project_status_v1'
LAYERS={'planned','implemented','deployed','running','accepted'}
STATES={'not_started','waiting','partial','documented','verified','not_rechecked','not_applicable'}
SCOPES={'source_review_only','runtime_and_evidence'}
REQUIRED={'schema','project','goal','phase','next_acceptance','dependencies','source_commit',
          'evidence_cutoff','verified_at','verification_scope','layers','results','limitations',
          'blockers','actions','decisions','evidence'}
RESULT_SCOPES={'accepted_matched_group','preliminary_single_seed','negative_result'}


def instant(value):
    t=datetime.fromisoformat(value.replace('Z','+00:00'))
    if t.tzinfo is None:
        raise ValueError('Timezone required')
    return t.timestamp()


def validate(x,project,now):
    if set(x)!=REQUIRED or x['schema']!=SCHEMA or x['project']!=project:
        raise ValueError('Invalid project status identity or fields')
    if not re.fullmatch('[0-9a-f]{40}',x['source_commit']):
        raise ValueError('Exact source commit required')
    if x['verification_scope'] not in SCOPES:
        raise ValueError('Unknown verification scope')
    for key in ('verified_at','evidence_cutoff'):
        if instant(x[key])>now+60:
            raise ValueError('Future evidence timestamp')
    if instant(x['evidence_cutoff'])>instant(x['verified_at']):
        raise ValueError('Evidence newer than verification')
    if set(x['layers'])!=LAYERS or any(v not in STATES for v in x['layers'].values()):
        raise ValueError('Invalid evidence layers')
    if x['verification_scope']=='source_review_only' and x['layers']['running']=='verified':
        raise ValueError('Repository review cannot certify live training')
    for key in ('dependencies','results','limitations','blockers','actions','decisions','evidence'):
        if not isinstance(x[key],list):
            raise ValueError('Expected a list: '+key)
    urls=set()
    for e in x['evidence']:
        if set(e)!={'label','url'} or not e['url'].startswith('https://github.com/'):
            raise ValueError('Use reviewed GitHub evidence links, not raw storage or signed URLs')
        if '?' in e['url'] or '@' in e['url']:
            raise ValueError('No credentials or query tokens in evidence')
        urls.add(e['url'])
    for row in x['results']:
        if set(row)!={'summary','scope','acceptance_url'} or row['scope'] not in RESULT_SCOPES:
            raise ValueError('Unsupported result claim')
        if row['acceptance_url'] not in urls:
            raise ValueError('Results require an explicit reviewed acceptance reference')
    return x


def main():
    root=Path(__file__).resolve().parents[1]
    payload=json.loads((root/'docs/handoff/status.json').read_text())
    validate(payload,'Radon_Bridge',datetime.now(timezone.utc).timestamp())
    print(json.dumps({'passed':True,'verification_scope':payload['verification_scope'],
                      'evidence_cutoff':payload['evidence_cutoff']},ensure_ascii=False))


if __name__=='__main__':
    main()
