"""CMX FRM+FFM A -> same frozen A continue / Radon / ordinary matched package.

Degeneracy is recorded as a coverage/interpretation label only.  It never
changes the registered stopping rule, selected checkpoint, seed, or whether the
same-A matched continuation is retained.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from radon_bridge.runtime.state import atomic_write_json
from radon_bridge.studies.cohort_case import sha
from radon_bridge.studies.cohort_augmentation import migrate
from radon_bridge.studies.cohort_attention_augmentation import (
    native_state_identity,fit_basis,validate_basis,
)

CASE_IDS=("host_continue","host_radon","host_linear")
COMPARISONS=[["host_radon","host_continue"],["host_linear","host_continue"],["host_radon","host_linear"]]
FIXED=dict(seed=3416,stage=3,channel_rank=32,M=32,S=64,kernel=3,group_count=1,
           real_batch=16,minimum_epochs=8,maximum_epochs=60,patience=6,min_delta=.001)


def read(path): return json.loads(Path(path).read_text())
def write(path,value): atomic_write_json(value,Path(path))


def _accepted_host(source_root):
    root=Path(source_root)
    q=read(root/"queue.json");status=read(root/"status.json")
    if q.get("study_kind")!="cmx_full_host" or q.get("test_used") is not False:
        raise ValueError("CMX-full host queue identity changed")
    if status.get("state")!="complete" or status.get("accepted")!=1 or status.get("failed") or status.get("test_used") is not False:
        raise ValueError("CMX-full host execution changed")
    case=q["cases"][0];trial=root/"trials"/case["name"];receipt=read(trial/"accepted.json");cfg=read(case["config"])
    if receipt.get("configuration")!=cfg or receipt.get("converged_by_policy") is not True or receipt.get("test_used") is not False:
        raise ValueError("CMX-full selected checkpoint identity changed")
    for name,digest in receipt["files"].items():
        if sha(trial/name)!=digest: raise ValueError("CMX-full artifact changed: "+name)
    return q,case,trial,receipt,cfg


def _selected_coverage(q,trial,receipt,cfg):
    native=native_state_identity(trial,cfg)
    state=torch.load(trial/"best.pt",map_location="cpu",weights_only=False)["model"]
    gates=[]
    for module,values in state.items():
        for name,value in values.items():
            if name=="return_gate" or name.endswith(".return_gate"):
                gates.extend(float(x) for x in value.detach().reshape(-1))
    reference=Path(q["references"][0]["trial"])/"selected_predictions.npz"
    changed=False
    with np.load(trial/"selected_predictions.npz",allow_pickle=False) as selected, np.load(reference,allow_pickle=False) as base:
        if not np.array_equal(selected["ids"],base["ids"]) or not np.array_equal(selected["y"],base["y"]):
            raise ValueError("CMX-full/reference participant identity changed")
        changed=any(not np.array_equal(selected[key],base[key]) for key in ("cfp","oct"))
    gate_nonzero=any(x!=0.0 for x in gates)
    nondegenerate=(not native["exact"]) and gate_nonzero and changed
    return dict(
        selected_epoch=receipt["best_epoch"],native_parent_state_identity=native,
        return_gate_values=gates,return_gate_nonzero=gate_nonzero,
        selected_predictions_differ_from_no_communication=changed,
        coverage_label="nondegenerate_project_cmx_full_A" if nondegenerate else "identity_or_degenerate_project_cmx_full_A",
        full_method_coverage_claim_allowed=nondegenerate,
        outcome_policy="label only; never alter stopping, selected checkpoint, seed, or repeat based on this label")


def prepare(source_root,output,sequence_id,source_commit,framework_commit,source_code_root):
    source_root=Path(source_root);out=Path(output);source_code_root=Path(source_code_root)
    if out.exists(): raise FileExistsError(out)
    q,case,trial,receipt,cfg=_accepted_host(source_root)
    out.mkdir(parents=True);(out/"configs").mkdir();(out/"trials").mkdir()
    migrate(trial,out/"host")
    coverage=_selected_coverage(q,trial,receipt,cfg)
    write(out/"coverage_identity.json",coverage)
    write(out/"host_basis_config.json",dict(
        host_checkpoint=dict(path=str((trial/"best.pt").resolve()),sha256=receipt["files"]["best.pt"])))
    protocol=dict(
        schema="radon_cmx_full_augmentation_protocol_v1",study_kind="existing_method_augmentation",
        sequence_id=sequence_id,source_commit=source_commit,framework_commit=framework_commit,
        source_cmx_full_root=str(source_root),source_cmx_full_status_sha256=sha(source_root/"status.json"),
        source_cmx_full_receipt_sha256=sha(trial/"accepted.json"),
        source_cmx_full_prediction_sha256=receipt["files"]["selected_predictions.npz"],
        source_cmx_full_best_sha256=receipt["files"]["best.pt"],source_cmx_full_best_epoch=receipt["best_epoch"],
        source_cmx_full_config=cfg,coverage=coverage,data=q["data"],test_used=False,
        fixed=FIXED,comparisons=COMPARISONS,
        basis_policy="fresh uncentered SVD from the exact selected CMX-full A train Stage3 pre-write features for every selected outcome, including best0 identity",
        host_semantics="pinned-author FRM+FFM communication-core project adaptation; not full author segmentation system",
        interpretation_boundary="degeneracy limits method-coverage claims only; the exact selected A is retained and used without outcome-driven retries",
        question="From the exact same selected CMX-full A, how do continuation, Radon addition, and ordinary communication differ?")
    write(out/"protocol.json",protocol)
    old_dep=read(source_root/"dependencies_acceptance.json")
    if old_dep.get("passed") is not True or old_dep.get("test_used") is not False:
        raise ValueError("CMX-full source dependency acceptance changed")
    refs={
        str(trial/"accepted.json"):sha(trial/"accepted.json"),
        str(trial/"best.pt"):receipt["files"]["best.pt"],
        str(trial/"selected_predictions.npz"):receipt["files"]["selected_predictions.npz"],
        str(out/"host/manifest.json"):sha(out/"host/manifest.json"),
        str(out/"host/model.pt"):sha(out/"host/model.pt"),
        str(out/"coverage_identity.json"):sha(out/"coverage_identity.json"),
    }
    write(out/"preparation_acceptance.json",dict(
        schema="radon_cmx_full_augmentation_preparation_v1",passed=True,test_used=False,
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


def finalize(root,source_code_root):
    root=Path(root);source_code_root=Path(source_code_root);protocol=read(root/"protocol.json");basis=validate_basis(root)
    if (root/"queue.json").exists(): return read(root/"queue.json")
    host=read(root/"host/manifest.json");base=copy.deepcopy(host["configuration"]);bases=basis["basis_files"]
    href=dict(path=str((root/"host/manifest.json").resolve()),sha256=sha(root/"host/manifest.json"))
    configs={};cases=[]
    for key,cfg in matched_configs(base,bases,href).items():
        path=root/"configs"/(cfg["name"]+".json");write(path,cfg);configs[str(path)]=sha(path)
        cases.append(dict(
            id=key,name=cfg["name"],config=str(path),resource=key,seed=3416,
            provenance="same exact selected CMX-full A; matched second-stage continuation independent of coverage label"))
    coverage=read(root/"coverage_identity.json")
    queue=dict(
        schema="radon_small_cohort_core_v1",study_kind="existing_method_augmentation",
        sequence_id=protocol["sequence_id"],data=protocol["data"],test_used=False,cases=cases,
        comparisons=COMPARISONS,
        host_summary=dict(
            method="CMX FRM+FFM communication-core project adapter",display_name="CMX-FRM+FFM-core",
            best_epoch=protocol["source_cmx_full_best_epoch"],
            source_model_sha256=protocol["source_cmx_full_best_sha256"],
            native_parent_state_exact=coverage["native_parent_state_identity"]["exact"],
            coverage_label=coverage["coverage_label"],
            full_method_coverage_claim_allowed=coverage["full_method_coverage_claim_allowed"],
            limitation_tag="project_cmx_full_core_not_author_segmentation_system",
            system_boundary="selected A retained exactly even if identity/degenerate; coverage label constrains interpretation, not training/selection",
            basis_policy=protocol["basis_policy"]))
    write(root/"queue.json",queue)
    prep=read(root/"preparation_acceptance.json");refs=dict(prep["references"])
    refs[str(root/"basis_fit/summary.json")]=sha(root/"basis_fit/summary.json")
    for ref in bases.values(): refs[ref["path"]]=ref["sha256"]
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
    a=sub.add_parser("prepare");a.add_argument("--source-cmx-full",required=True);a.add_argument("--output",required=True)
    a.add_argument("--sequence-id",required=True);a.add_argument("--source-commit",required=True)
    a.add_argument("--framework-commit",required=True);a.add_argument("--source-root",required=True)
    a=sub.add_parser("fit-basis");a.add_argument("--root",required=True)
    a=sub.add_parser("finalize");a.add_argument("--root",required=True);a.add_argument("--source-root",required=True)
    x=p.parse_args()
    if x.cmd=="prepare":result=prepare(x.source_cmx_full,x.output,x.sequence_id,x.source_commit,x.framework_commit,x.source_root)
    elif x.cmd=="fit-basis":result=fit_basis(x.root)
    else:result=finalize(x.root,x.source_root)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":main()
