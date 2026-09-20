"""Finite CMX FRM+FFM communication-core A on the accepted WS02 small cohort."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

from radon_bridge.runtime.state import atomic_write_json
from radon_bridge.studies.cohort_case import sha
from radon_bridge.studies.cohort_cmx_host import AUTHOR, _accepted_core


def read(path): return json.loads(Path(path).read_text())
def write(path,value): atomic_write_json(value,Path(path))


def cmx_full_config_from_none(cfg,alignment_tokens=64,heads=4):
    value=copy.deepcopy(cfg)
    value["name"]="cmx_full_seed3416"
    value["bridges"]=[dict(
        nodes=["cfp_stage3","oct_stage3"],family="cmx_full",
        alignment_tokens=int(alignment_tokens),heads=int(heads))]
    return value


def prepare(core_root,output,source_root,source_commit,framework_commit,sequence_id,
            alignment_tokens=64,heads=4):
    core=Path(core_root);out=Path(output);source_root=Path(source_root)
    if out.exists(): raise FileExistsError(out)
    q,none,none_cfg,none_trial,none_receipt=_accepted_core(core)
    out.mkdir(parents=True);(out/"configs").mkdir()
    cfg=cmx_full_config_from_none(none_cfg,alignment_tokens,heads)
    cfg_path=out/"configs/cmx_full_seed3416.json";write(cfg_path,cfg)

    reference=dict(
        id="none",name=none["name"],config=none["config"],trial=str(none_trial),
        config_sha256=sha(none["config"]),receipt_sha256=sha(none_trial/"accepted.json"),
        resource="none",seed=3416,
        provenance="strict reuse of independently accepted no-communication reference",
        source_sequence_id=q["sequence_id"],source_id="none")
    case=dict(
        id="cmx_full",name="cmx_full_seed3416",config=str(cfg_path),
        resource="cmx_full",seed=3416,
        provenance="single-seed project adaptation of pinned CMX FRM+FFM communication core")
    queue=dict(
        schema="radon_small_cohort_core_v1",study_kind="cmx_full_host",
        sequence_id=sequence_id,source_commit=source_commit,framework_commit=framework_commit,
        author=AUTHOR,data=q["data"],test_used=False,references=[reference],cases=[case],
        comparisons=[["cmx_full","none"]],
        adaptation=dict(
            component="FeatureRectifyModule+FeatureFusionModule communication core",
            stage=3,alignment_tokens=int(alignment_tokens),
            alignment_shape=[int(alignment_tokens**0.5),int(alignment_tokens**0.5)],
            heads=int(heads),order="align native features first -> author FRM -> author FFM -> project branch return",
            source_fidelity="pinned author net_utils.py weights/forward/input-gradient/parameter-gradient parity",
            branch_return="project-only zero-initialized two-branch scalar return gates",
            boundary="project classifier communication-core adaptation; not full CMX segmentation-system reproduction",
            outcome_policy="use the registered stop/checkpoint policy exactly; selected best0/best>0 is retained without outcome-driven retry"))
    write(out/"queue.json",queue)

    source_deps=read(core/"dependencies_acceptance.json")
    if source_deps.get("passed") is not True or source_deps.get("test_used") is not False:
        raise ValueError("Source dependency acceptance changed")
    refs={
        str(none_trial/"accepted.json"):sha(none_trial/"accepted.json"),
        str(none_trial/"best.pt"):none_receipt["files"]["best.pt"],
        str(none_trial/"selected_predictions.npz"):none_receipt["files"]["selected_predictions.npz"],
        str(core/"independent_final_audit.json"):sha(core/"independent_final_audit.json"),
        str(core/"status.json"):sha(core/"status.json"),
    }
    deps=dict(
        passed=True,test_used=False,queue_sha256=sha(out/"queue.json"),
        config_files={str(cfg_path):sha(cfg_path)},references=refs,
        data_files=source_deps["data_files"])
    write(out/"dependencies_acceptance.json",deps)

    framework=read(source_root/"framework.lock.json")
    if framework["upstream_commit"]!=framework_commit: raise ValueError("Framework commit changed")
    files={}
    for p in sorted((source_root/"src/radon_bridge").rglob("*.py")):
        files[str(p.resolve())]=sha(p)
    for item in framework["files"]:
        p=source_root/item["local_path"]
        if sha(p)!=item["sha256"]: raise ValueError("Framework file changed")
        files[str(p.resolve())]=item["sha256"]
    actual=subprocess.check_output(["git","-C",str(source_root),"rev-parse","HEAD"],text=True).strip()
    if actual!=source_commit: raise ValueError("Source commit changed")
    write(out/"code_acceptance.json",dict(
        passed=True,passed_cpu=True,test_used=False,source_commit=source_commit,
        framework_commit=framework_commit,author=AUTHOR,files=files))
    write(out/"preparation_acceptance.json",dict(
        schema="radon_cmx_full_host_preparation_v1",passed=True,test_used=False,
        queue_sha256=sha(out/"queue.json"),
        dependencies_sha256=sha(out/"dependencies_acceptance.json"),
        code_sha256=sha(out/"code_acceptance.json"),
        source_core_status_sha256=sha(core/"status.json"),
        source_core_audit_sha256=sha(core/"independent_final_audit.json"),
        author=AUTHOR))
    return queue


def main():
    p=argparse.ArgumentParser();p.add_argument("--core",required=True);p.add_argument("--output",required=True)
    p.add_argument("--source-root",required=True);p.add_argument("--source-commit",required=True)
    p.add_argument("--framework-commit",required=True);p.add_argument("--sequence-id",required=True)
    p.add_argument("--alignment-tokens",type=int,default=64);p.add_argument("--heads",type=int,default=4)
    a=p.parse_args()
    print(json.dumps(prepare(a.core,a.output,a.source_root,a.source_commit,a.framework_commit,
        a.sequence_id,a.alignment_tokens,a.heads),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
