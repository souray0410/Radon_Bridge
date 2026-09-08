"""Read-only manifest lineage and duplicate-content audit; public aggregates only."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    lock=a.root/'unified_test_lock_v6';c=json.loads((lock/'candidate_lock.json').read_text())
    root=Path('/data/mengh/LOOK/2026_09_03_19_35_04/dataset')
    bank=root/'cohorts/task_scout';raw_labels=bank/'glaucoma_all_evidence/primary/reference_labels.csv'
    candidate=root/'phenotypes/record_phenotype_candidates.csv'
    table=lambda p:pd.read_csv(p,dtype=str).fillna('')
    old=Path(c['development_data_directory']);new=Path(c['data_directory'])
    cache=pd.concat([table(old/'selected.csv'),table(new/'selected.csv')],ignore_index=True).set_index('participant_id')
    ref=table(raw_labels).set_index('participant_id');assert set(cache.index)==set(ref.index)
    common=[k for k in ref.columns if k in cache];ref=ref.loc[cache.index]
    assert (cache[common]==ref[common]).all().all()
    acceptance=json.loads((lock/'data_acceptance.json').read_text());assert sha(raw_labels)==acceptance['labels_sha256']
    candidates=table(candidate);manifest=json.loads((bank/'task_bank_manifest.json').read_text())
    assert sha(candidate)==manifest['source_candidate_sha256']
    orig=candidates.set_index('participant_id').loc[cache.index]
    assert (cache.competing_eye_condition.str.lower()==orig.competing_eye_condition.str.lower()).all()
    assert (cache.phenotype_profile=='ukb_record_glaucoma_all_evidence_binary_bilateral').all()
    assert (orig.phenotype_profile=='ukb_record_prevalent_4class_bilateral').all()
    commonraw=[k for k in orig.columns if k in cache and k not in ('split','competing_eye_condition','phenotype_profile')]
    assert (cache[commonraw]==orig[commonraw]).all().all()
    split=table(bank/'participant_split_manifest.csv').set_index('participant_id').loc[cache.index]['split'];assert (cache.split==split).all()
    positive=cache[cache.label_id=='1'];negative=cache[cache.label_id=='0']
    eligible=candidates[(candidates.candidate_status=='prevalent_case')&(candidates.candidate_label=='glaucoma')]
    assert set(eligible.participant_id)==set(positive.index)
    assert (positive.candidate_status=='prevalent_case').all() and (positive.candidate_label=='glaucoma').all()
    assert (negative.candidate_status=='strict_control').all()
    assert (negative[['prevalent_targets','incident_targets','undated_targets']]=='').all().all()
    assert positive.prevalent_targets.map(lambda x:'glaucoma' in x.split(';')).all()
    hashes={k:collections.defaultdict(list) for k in ['cfp_eye','oct_eye','paired_participant']}
    native=collections.defaultdict(collections.Counter);missing=collections.Counter()
    for _,r in cache.iterrows():
        p=new if r.split=='test' else old
        with np.load(p/(r.order+'.npz'),allow_pickle=False) as z:
            cfp=np.load(p/'cfp224'/(r.order+'.npy'),allow_pickle=False);oct_=z['oct']
            for k,v in [('cfp_eye',cfp),('oct_eye',oct_)]:
                for eye in v:hashes[k][hashlib.sha256(eye.tobytes()).hexdigest()].append(r.split)
            hashes['paired_participant'][hashlib.sha256(cfp.tobytes()+oct_.tobytes()).hexdigest()].append(r.split)
            if 'native_sizes' in z:
                for s in z['native_sizes']:native[r.split]['x'.join(map(str,s))]+=1
            else:missing[r.split]+=2
    dup={k:dict(total_records=sum(len(x) for x in h.values()),duplicate_content_groups=sum(len(x)>1 for x in h.values()),cross_split_content_groups=sum(len(set(x))>1 for x in h.values())) for k,h in hashes.items()}
    m=manifest['profiles']['glaucoma_all_evidence']
    codes=Path('/home/mengh/LOOK/2026_09_03_19_35_04/tool/look_core')
    report=dict(state='complete',cache_equals_original_labels_all_shared_fields=True,cache_equals_candidate_fields_after_documented_transforms=True,documented_transforms={'competing_eye_condition':'false to False string capitalization; verified after lowercasing','phenotype_profile':'multi-disease candidate profile to glaucoma binary task profile, as implemented in task_scout._label_rows'},global_split_manifest_matches=True,participant_disjoint=True,positive_prevalent_rule_verified=True,negative_no_target_evidence_verified=True,
        candidate_pool_records=len(candidates),all_eligible_glaucoma_cases_included=True,candidate_status_counts=candidates.candidate_status.value_counts().to_dict(),selected_prevalent_cases=len(positive),selected_controls=len(negative),case_selection_rule=m['case_rule'],matching=m['matching'],
        content_duplicates=dup,native_dimension_metadata_observed=dict(native),native_dimension_metadata_missing_eyes=dict(missing),
        source_hashes={str(p):sha(p) for p in [raw_labels,candidate,bank/'task_bank_manifest.json',bank/'participant_split_manifest.csv',codes/'task_scout.py',codes/'cohort.py',codes/'phenotype.py']},
        limits=['Record-derived labels verified against archived candidate records; not an independent re-extraction of raw UKB clinical fields or expert relabeling.','Duplicate hashes exclude exact content duplicates only, not near duplicates or biological relatedness.','Native dimension metadata absent in older caches; these entries are not inferred as measured native shapes.'])
    (a.output/'lineage_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['state','candidate_pool_records','selected_prevalent_cases','selected_controls','content_duplicates','native_dimension_metadata_missing_eyes']}),flush=True)

if __name__=='__main__':main()
