"""Prepare one finite CMX-FRM A on the accepted small cohort.

This trains the project cross-dimensional adaptation of the author Feature
Rectify Module as an A.  It deliberately does not start the matched A+bridge
second stage; that is created only after this A has an accepted checkpoint.
"""
from __future__ import annotations
import argparse, copy, json, subprocess
from pathlib import Path

from radon_bridge.runtime.state import atomic_write_json
from radon_bridge.studies.cohort_case import sha


AUTHOR = dict(
    repository="huaaaliu/RGBX_Semantic_Segmentation",
    commit="e251d860aebc2f583a6c4919877e6bebe7f1aff3",
    license="MIT",
    license_sha256="a3fb69f7d2d7ab44ce80bba7f8c3a61f3c8a2775a2baac4bef815a60c4d8ba5e",
    net_utils_sha256="ada5e36e14d83c35d9230618c4eb84352a76ec2275a086a624b280e7638d8473",
)


def read(path): return json.loads(Path(path).read_text())
def write(path,value): atomic_write_json(value,Path(path))


def _accepted_core(core):
    core=Path(core);q=read(core/"queue.json");status=read(core/"status.json");audit=read(core/"independent_final_audit.json")
    if q.get("schema")!="radon_small_cohort_core_v1" or q.get("test_used") is not False:
        raise ValueError("Accepted small-cohort queue changed")
    if status.get("state")!="complete" or status.get("accepted")!=6 or status.get("failed") or status.get("test_used") is not False:
        raise ValueError("Accepted small-cohort execution changed")
    if audit.get("passed") is not True or audit.get("test_used") is not False:
        raise ValueError("Accepted small-cohort independent audit changed")
    none=next((x for x in q["cases"] if x["id"]=="none"),None)
    if none is None: raise ValueError("Accepted no-communication reference missing")
    cfg=read(none["config"]);trial=core/"trials"/none["name"];receipt=read(trial/"accepted.json")
    if receipt.get("configuration")!=cfg or receipt.get("converged_by_policy") is not True or receipt.get("test_used") is not False:
        raise ValueError("No-communication reference changed")
    for name,digest in receipt["files"].items():
        if sha(trial/name)!=digest: raise ValueError("No-communication artifact changed: "+name)
    if cfg.get("bridges")!=[] or cfg.get("seed")!=3416:
        raise ValueError("No-communication reference configuration changed")
    return q,none,cfg,trial,receipt


def cmx_config_from_none(cfg,alignment_tokens=64):
    value=copy.deepcopy(cfg)
    value["name"]="cmx_frm_seed3416"
    value["bridges"]=[dict(nodes=["cfp_stage3","oct_stage3"],family="cmx_frm",alignment_tokens=int(alignment_tokens))]
    return value


def prepare(core_root,output,source_root,source_commit,framework_commit,alignment_tokens=64):
    core=Path(core_root);out=Path(output);source_root=Path(source_root)
    if out.exists(): raise FileExistsError(out)
    q,none,none_cfg,none_trial,none_receipt=_accepted_core(core)
    out.mkdir(parents=True);(out/"configs").mkdir()
    cfg=cmx_config_from_none(none_cfg,alignment_tokens)
    cfg_path=out/"configs/cmx_frm_seed3416.json";write(cfg_path,cfg)

    reference=dict(id="none",name=none["name"],config=none["config"],trial=str(none_trial),
        config_sha256=sha(none["config"]),receipt_sha256=sha(none_trial/"accepted.json"),
        resource="none",seed=3416,provenance="strict reuse of independently accepted no-communication reference",
        source_sequence_id=q["sequence_id"],source_id="none")
    case=dict(id="cmx_frm",name="cmx_frm_seed3416",config=str(cfg_path),resource="cmx_frm",seed=3416,
        provenance="new single-seed project adaptation of CMX Feature Rectify Module")
    queue=dict(schema="radon_small_cohort_core_v1",study_kind="cmx_frm_host",
        sequence_id="cmx_frm_teacher_20260919_v1",source_commit=source_commit,framework_commit=framework_commit,
        author=AUTHOR,data=q["data"],test_used=False,references=[reference],cases=[case],
        comparisons=[["cmx_frm","none"]],
        adaptation=dict(component="FeatureRectifyModule",stage=3,alignment_tokens=int(alignment_tokens),
            equal_grid_parity="author FRM",cross_dimensional_spatial_rule="fixed shared flattened-token lattice with deterministic linear resampling",
            boundary="project component adaptation; not full CMX segmentation reproduction"))
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
    deps=dict(passed=True,test_used=False,queue_sha256=sha(out/"queue.json"),
        config_files={str(cfg_path):sha(cfg_path)},references=refs,data_files=source_deps["data_files"])
    write(out/"dependencies_acceptance.json",deps)

    framework=read(source_root/"framework.lock.json")
    if framework["upstream_commit"]!=framework_commit: raise ValueError("Framework commit changed")
    files={}
    for p in sorted((source_root/"src/radon_bridge").rglob("*.py")): files[str(p.resolve())]=sha(p)
    for item in framework["files"]:
        p=source_root/item["local_path"]
        if sha(p)!=item["sha256"]: raise ValueError("Framework file changed")
        files[str(p.resolve())]=item["sha256"]
    actual=subprocess.check_output(["git","-C",str(source_root),"rev-parse","HEAD"],text=True).strip()
    if actual!=source_commit: raise ValueError("Source commit changed")
    write(out/"code_acceptance.json",dict(passed=True,passed_cpu=True,test_used=False,
        source_commit=source_commit,framework_commit=framework_commit,author=AUTHOR,files=files))
    write(out/"preparation_acceptance.json",dict(schema="radon_cmx_frm_host_preparation_v1",passed=True,test_used=False,
        queue_sha256=sha(out/"queue.json"),dependencies_sha256=sha(out/"dependencies_acceptance.json"),
        code_sha256=sha(out/"code_acceptance.json"),source_core_status_sha256=sha(core/"status.json"),
        source_core_audit_sha256=sha(core/"independent_final_audit.json"),author=AUTHOR))
    return queue


def main():
    p=argparse.ArgumentParser();p.add_argument("--core",required=True);p.add_argument("--output",required=True)
    p.add_argument("--source-root",required=True);p.add_argument("--source-commit",required=True);p.add_argument("--framework-commit",required=True)
    p.add_argument("--alignment-tokens",type=int,default=64)
    a=p.parse_args();value=prepare(a.core,a.output,a.source_root,a.source_commit,a.framework_commit,a.alignment_tokens)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
