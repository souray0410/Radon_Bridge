import importlib.util
from pathlib import Path

def test_repository_contract():
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location("research_manage",root/"scripts/manage.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.check()["structure"]
