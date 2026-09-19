"""Finite CMX-FRM A -> same-A continue / Radon / ordinary matched package.

The accepted CMX-FRM A selected epoch0 identity. This package deliberately
preserves that fact: all three second-stage arms start from the exact same
accepted checkpoint. The result is therefore a degenerate/negative CMX host
coverage case, not evidence that a nontrivial CMX communication state existed.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path

from radon_bridge.runtime.state import atomic_write_json
from radon_bridge.studies.cohort_case import sha
from radon_bridge.studies.cohort_augmentation import migrate
from radon_bridge.studies.cohort_attention_augmentation import (
    native_state_identity,
    fit_basis,
    validate_basis,
)

CASE_IDS=("host_continue","host_radon","host_linear")
COMPARISONS=[["host_radon","host_continue"],["host_linear","host_continue"],["host_radon","host_linear"]]
FIXED=dict(seed=3416,stage=3,channel_rank=32,M=32,S=64,kernel=3,group_count=1,
           real_batch=16,minimum_epochs=8,maximum_epochs=60,patience=6,min_delta=.001)


def read(path): return json.loads(Path(path).read_text())
def write(path,value): atomic_write_json(value,Path(path))


def _accepted_cmx(source_root):
    root=Path(source_root)
    q=read(root/"queue.json");status=read(root/"status.json");audit=read(root/"independent_final_audit.json")
    if q.get("study_kind")!="cmx_frm_host" or q.get("test_used") is not False:
        raise ValueError("CMX host queue identity changed")
    if status.get("state")!="complete" or status.get("accepted")!=1 or status.get("failed") or status.get("test_used") is not False:
        raise ValueError("CMX host execution changed")
    if (audit.get("passed") is not True or audit.get("test_used") is not False
            or audit.get("selected_epoch")!=0
            or audit.get("selected_predictions_exact_no_communication_reference") is not True
            or audit.get("residual_gate_zero_at_selected_checkpoint") is not True):
        raise ValueError("CMX independent audit changed")
    case=q["cases"][0];trial=root/"trials"/case["name"];receipt=read(trial/"accepted.json");cfg=read(case["config"])
    if (receipt.get("configuration")!=cfg or receipt.get("converged_by_policy") is not True
            or receipt.get("best_epoch")!=0 or receipt.get("test_used") is not False):
        raise ValueError("CMX selected checkpoint identity changed")
    for name,digest in receipt["files"].items():
        if sha(trial/name)!=digest: raise ValueError("CMX artifact changed: "+name)
    return q,case,trial,receipt,cfg,audit


def prepare(source_root, output, sequence_id, source_commit, framework_commit, source_code_root):
    source_root=Path(source_root);out=Path(output);source_code_root=Path(source_code_root)
    if out.exists(): raise FileExistsError(out)
    q,case,trial,receipt,cfg,audit=_accepted_cmx(source_root)
    out.mkdir(parents=True);(out/"configs").mkdir();(out/"trials").mkdir()
    migrate(trial,out/"host")
    native=native_state_identity(trial,cfg)
    # Epoch0 identity is expected and scientifically material, not an error.
    if native["exact"] is not True:
        raise ValueError("Selected CMX epoch0 unexpectedly changed native parents")
    write(out/"host_basis_config.json",dict(
        host_checkpoint=dict(path=str((trial/"best.pt").resolve()),sha256=receipt["files"]["best.pt"])))
    protocol=dict(
        schema="radon_cmx_augmentation_protocol_v1",study_kind="existing_method_augmentation",
        sequence_id=sequence_id,source_commit=source_commit,framework_commit=framework_commit,
        source_cmx_root=str(source_root),source_cmx_status_sha256=sha(source_root/"status.json"),
        source_cmx_audit_sha256=sha(source_root/"independent_final_audit.json"),
        source_cmx_receipt_sha256=sha(trial/"accepted.json"),
        source_cmx_prediction_sha256=receipt["files"]["selected_predictions.npz"],
        source_cmx_best_sha256=receipt["files"]["best.pt"],source_cmx_best_epoch=0,
        source_cmx_config=cfg,native_parent_state_identity=native,
        data=q["data"],test_used=False,fixed=FIXED,comparisons=COMPARISONS,
        basis_policy="fresh uncentered SVD from exact accepted CMX best0 train Stage3 pre-write features",
        host_semantics="CMX FeatureRectifyModule project adaptation; selected checkpoint is exact identity/no-communication state; not full author CMX system",
        interpretation_boundary="three arms are matched second-stage continuations from the same accepted CMX-init checkpoint; they do not establish that the incoming A was nontrivial",
        question="From the exact same accepted CMX-init checkpoint, how do continuation, Radon addition, and ordinary communication differ?")
    write(out/"protocol.json",protocol)
    old_dep=read(source_root/"dependencies_acceptance.json")
    if old_dep.get("passed") is not True or old_dep.get("test_used") is not False:
        raise ValueError("CMX source dependency acceptance changed")
    refs={
        str(trial/"accepted.json"):sha(trial/"accepted.json"),
        str(trial/"best.pt"):receipt["files"]["best.pt"],
        str(trial/"selected_predictions.npz"):receipt["files"]["selected_predictions.npz"],
        str(source_root/"independent_final_audit.json"):sha(source_root/"independent_final_audit.json"),
        str(out/"host/manifest.json"):sha(out/"host/manifest.json"),
        str(out/"host/model.pt"):sha(out/"host/model.pt"),
    }
    write(out/"preparation_acceptance.json",dict(
        schema="radon_cmx_augmentation_preparation_v1",passed=True,test_used=False,
        protocol_sha256=sha(out/"protocol.json"),host_basis_config_sha256=sha(out/"host_basis_config.json"),
        references=refs,data_files=old_dep["data_files"]))
    return protocol


def matched_configs(base,bases,href):
    result={}
    for key,mode in (("host_continue",None),("host_radon","radon"),("host_linear","linear_resample")):
        cfg=copy.deepcopy(base);cfg["name"]=key+"_seed3416";cfg["augmentation_host"]=copy.deepcopy(href)
        if mode:
            cfg["bridges"].append(dict(
                nodes=["cfp_stage3","oct_stage3"],M=32,S=64,rho=.125,mode=mode,
                compression="fixed_svd_channel",basis_files=copy.deepcopy(bases),kernel_size=3,parallel_to=0))
        result[key]=cfg
    return result


def finalize(root, source_code_root):
    root=Path(root);source_code_root=Path(source_code_root);protocol=read(root/"protocol.json");basis=validate_basis(root)
    if (root/"queue.json").exists(): return read(root/"queue.json")
    host=read(root/"host/manifest.json");base=copy.deepcopy(host["configuration"]);bases=basis["basis_files"]
    href=dict(path=str((root/"host/manifest.json").resolve()),sha256=sha(root/"host/manifest.json"))
    configs={};cases=[]
    for key,cfg in matched_configs(base,bases,href).items():
        path=root/"configs"/(cfg["name"]+".json");write(path,cfg);configs[str(path)]=sha(path)
        cases.append(dict(id=key,name=cfg["name"],config=str(path),resource=key,seed=3416,
            provenance="same accepted CMX-init best0 checkpoint; matched second-stage continuation"))
    queue=dict(
        schema="radon_small_cohort_core_v1",study_kind="existing_method_augmentation",
        sequence_id=protocol["sequence_id"],data=protocol["data"],test_used=False,cases=cases,
        comparisons=COMPARISONS,
        host_summary=dict(
            method="CMX-FRM project adapter (selected best0 identity)",
            display_name="CMX-FRM-init",
            best_epoch=0,source_model_sha256=protocol["source_cmx_best_sha256"],
            native_parent_state_exact=True,
            limitation_tag="project_cmx_frm_component_adapter_selected_identity_checkpoint",
            system_boundary="项目CMX-FRM组件适配；A选中epoch0恒等状态，不声称复现作者完整CMX，也不声称incoming A已形成非退化通信",
            basis_policy=protocol["basis_policy"]))
    write(root/"queue.json",queue)
    prep=read(root/"preparation_acceptance.json");refs=dict(prep["references"])
    refs[str(root/"basis_fit/summary.json")]=sha(root/"basis_fit/summary.json")
    for ref in bases.values():refs[ref["path"]]=ref["sha256"]
    write(root/"dependencies_acceptance.json",dict(
        passed=True,test_used=False,queue_sha256=sha(root/"queue.json"),
        config_files=configs,references=refs,data_files=prep["data_files"],
        preparation_acceptance_sha256=sha(root/"preparation_acceptance.json")))
    framework=read(source_code_root/"framework.lock.json");files={}
    for p in sorted((source_code_root/"src/radon_bridge").rglob("*.py")):files[str(p.resolve())]=sha(p)
    for item in framework["files"]:
        p=source_code_root/item["local_path"]
        if sha(p)!=item["sha256"]:raise ValueError("Framework source lock changed")
        files[str(p.resolve())]=item["sha256"]
    write(root/"code_acceptance.json",dict(
        passed=True,passed_cpu=True,test_used=False,
        source_commit=protocol["source_commit"],framework_commit=protocol["framework_commit"],files=files))
    return queue


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("prepare");a.add_argument("--source-cmx",required=True);a.add_argument("--output",required=True);a.add_argument("--sequence-id",required=True);a.add_argument("--source-commit",required=True);a.add_argument("--framework-commit",required=True);a.add_argument("--source-root",required=True)
    a=sub.add_parser("fit-basis");a.add_argument("--root",required=True)
    a=sub.add_parser("finalize");a.add_argument("--root",required=True);a.add_argument("--source-root",required=True)
    x=p.parse_args()
    if x.cmd=="prepare":result=prepare(x.source_cmx,x.output,x.sequence_id,x.source_commit,x.framework_commit,x.source_root)
    elif x.cmd=="fit-basis":result=fit_basis(x.root)
    else:result=finalize(x.root,x.source_root)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":main()
