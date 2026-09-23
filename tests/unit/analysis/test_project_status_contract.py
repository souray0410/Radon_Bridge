import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import pytest

from scripts.check_project_status_contract import REQUIRED, validate


ROOT=Path(__file__).resolve().parents[3]
STATUS=ROOT/'docs/handoff/status.json'


def payload():
    return json.loads(STATUS.read_text())


def now():
    return datetime.now(timezone.utc).timestamp()


def test_current_status_matches_shared_phd_contract():
    value=payload()
    assert set(value)==REQUIRED
    assert value['verification_scope']=='runtime_and_evidence'
    assert value['evidence_cutoff']=='2026-09-23T17:13:00+00:00'
    assert validate(value,'Radon_Bridge',now()) is value


def test_unknown_verification_scope_is_rejected():
    value=copy.deepcopy(payload());value['verification_scope']='free_form_explanation'
    with pytest.raises(ValueError,match='Unknown verification scope'):
        validate(value,'Radon_Bridge',now())


def test_result_requires_reviewed_evidence_url():
    value=copy.deepcopy(payload());value['results'][0]['acceptance_url']='https://github.com/example/missing'
    with pytest.raises(ValueError,match='explicit reviewed acceptance reference'):
        validate(value,'Radon_Bridge',now())
