import json
from pathlib import Path
import pytest
from radon_bridge.artifacts import resolve,relocate
from radon_bridge.workspace import Workspace

def test_artifacts_and_mapping_are_independent_of_cwd(tmp_path,monkeypatch):
    root=tmp_path/"project with spaces";root.mkdir()
    monkeypatch.setenv("RB_PROJECT_ROOT",str(root))
    monkeypatch.delenv("RB_ARTIFACT_MAP",raising=False)
    item=root/"basis.pt";item.write_bytes(b"synthetic")
    first=resolve("basis.pt");monkeypatch.chdir(tmp_path)
    assert resolve("basis.pt")==first==item
    old=tmp_path/"old"
    mapping={"schema":"artifact_path_map_v1","mappings":[{"source":str(old),"target":"archive"},{"source":str(old/"current"),"target":"special"}]}
    file=root/"map.json";file.write_text(json.dumps(mapping));before=file.read_bytes()
    monkeypatch.setenv("RB_ARTIFACT_MAP","map.json")
    record={"path":str(old/"current"/"model.pt"),"metric":"unchanged"}
    result=relocate(record)
    assert result["path"]==str(root/"special/model.pt")
    assert record["path"]==str(old/"current/model.pt")
    assert file.read_bytes()==before
    old.mkdir();existing=old/"keep.pt";existing.write_bytes(b"keep")
    assert resolve(existing)==existing

def test_no_implicit_historical_server_map(tmp_path,monkeypatch):
    monkeypatch.delenv("RB_ARTIFACT_MAP",raising=False)
    nonexistent=tmp_path/"old_unmapped.pt"
    assert resolve(nonexistent)==nonexistent

@pytest.mark.parametrize("schema",["research_deployment_v1","radon_bridge_workspace_v1"])
def test_workspace_relative_roots_use_project_not_cwd(tmp_path,monkeypatch,schema):
    root=tmp_path/"project";root.mkdir();(root/"project.json").write_text("{}")
    cfg=root/"configs/deployment";cfg.mkdir(parents=True)
    file=cfg/"local.json";file.write_text(json.dumps({"schema":schema,"code_root":"code","data_root":"data","study":"2026_09_09_10_30_34","project":"Radon_Bridge"}))
    first=Workspace.load(file);monkeypatch.chdir(tmp_path)
    assert Workspace.load(file)==first
    assert first.code_root==root/"code" and first.data_root==root/"data"
    assert not first.data_root.exists()
