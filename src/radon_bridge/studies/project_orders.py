"""Publish full paired-study cases only when all native repetitions are accepted."""
from pathlib import Path
from radon_bridge.models.native_materialization import materialize_selected
from radon_bridge.runtime.state import atomic_write_json,file_sha256,stable_hash
from radon_bridge.studies.native_prerequisites import MODELS,DISEASES
from radon_bridge.studies.autoresearch import read,immutable
from radon_bridge.studies.research_matrix import arms,coverage


def advance(config,groups,verify_native,reserve):
    p=config['project'];root=Path(p['output']);root.mkdir(parents=True,exist_ok=True);tasks=[];states={}
    gate=p['runtime_gate']
    if file_sha256(Path(gate['path']))!=gate['sha256'] or read(gate['path']).get('status')!='accepted':raise ValueError('Actual runtime integration not accepted')
    for disease in DISEASES:
        for architecture in sorted({r['model'] for r in read(config['catalog']['path'])['candidates']}):
            name=disease+'/'+architecture
            pair=[groups[name+'/'+track] for track in ('cfp_2d','oct_volume_3d')]
            if not all(g['state']=='waiting_project_adapter' for g in pair):states[name]='waiting_paired_three_seed_parents';continue
            selected={}
            for role,group in zip(('cfp','oct'),pair):
                selected[role]={}
                for source in [group['selected']['run_dir']]+[r['run_dir'] for r in group['replicas']]:
                    spec=read(Path(source)/'spec.json');seed=spec['training']['seed']
                    destination=materialize_selected(source,root/'parents',spec,verify_native)
                    selected[role][seed]=dict(path=str(destination),manifest_sha256=file_sha256(destination/'selected_artifact.json'))
            for seed in (3416,3417,3418):
                spec=dict(schema='radon_project_case_v1',model={'name':architecture},disease=disease,seed=seed,
                    parents={role:selected[role][seed] for role in selected},training=p['training'],arms=arms(disease,architecture),
                    bootstrap_iterations=10000,source_pins=config['source_pins'],test_access=False,
                    protocol_sha256=config['protocol']['sha256'],catalog_sha256=config['catalog']['sha256'])
                key=name+f'/seed{seed}'
                path=root/'specs'/(stable_hash(key)+'.json');path.parent.mkdir(exist_ok=True);immutable(path,spec)
                run=reserve(root,'radon_expanded_'+config['catalog']['sha256'][:16],key,spec,
                    source={'protocol':config['protocol'],'native_groups':name},refresh_summary=False)
                tasks.append(dict(id=key,spec=str(path),spec_sha256=file_sha256(path),run_dir=str(run),role='radon_project'))
            states[name]='project_tasks_registered'
    queue=dict(schema='radon_bridge_project_work_feed_v1',tasks=tasks,test_access=False)
    atomic_write_json(queue,root/'queue.json');atomic_write_json(coverage(),root/'coverage.json')
    from radon_bridge.analysis.project_rollup import summarize
    report=summarize(tasks,root/'report')
    return dict(tasks=len(tasks),accepted=report['accepted'],complete=report['complete'],groups=states,
        planned_training_positions=sum(len(arms(d,m)) for d in DISEASES for m in {r['model'] for r in read(config['catalog']['path'])['candidates']})*3,queue=str(root/'queue.json'))
