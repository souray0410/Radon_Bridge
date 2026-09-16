"""Publish the first accepted seed without waiting for unrelated replications."""
from pathlib import Path
from radon_bridge.models.native_materialization import materialize_selected
from radon_bridge.runtime.state import atomic_write_json,file_sha256,stable_hash
from radon_bridge.studies.native_prerequisites import MODELS,DISEASES
from radon_bridge.studies.autoresearch import read,immutable
from radon_bridge.studies.research_matrix import arms,coverage


def pilot_accepted(root, name):
    """Require the owning complete-case verifier, not a score threshold or PID."""
    path=Path(root)/'bindings'/(stable_hash(name+'/seed3416')+'.json')
    if not path.exists():return False
    binding=read(path)
    if file_sha256(Path(binding['spec']))!=binding['spec_sha256']:
        raise ValueError('Pilot specification changed')
    spec=read(binding['spec']);run=Path(binding['run_dir'])
    if not (run/'accepted.json').exists():return False
    from radon_bridge.studies.project_case import verify_case
    verify_case(run,spec)
    return True


def advance(config,groups,verify_native,reserve):
    p=config['project'];root=Path(p['output']);root.mkdir(parents=True,exist_ok=True);tasks=[];states={}
    gate=p['runtime_gate']
    if file_sha256(Path(gate['path']))!=gate['sha256'] or read(gate['path']).get('status')!='accepted':raise ValueError('Actual runtime integration not accepted')
    for disease in DISEASES:
        for architecture in sorted({r['model'] for r in read(config['catalog']['path'])['candidates']}):
            name=disease+'/'+architecture
            pair=[groups.get(name+'/'+track,{}) for track in ('cfp_2d','oct_volume_3d')]
            if not all(g.get('selected') for g in pair):
                states[name]='waiting_locked_parent_selection';continue
            selected={}
            for role,group in zip(('cfp','oct'),pair):
                selected[role]={}
                sources=[group['selected']['run_dir']]+[r['run_dir'] for r in group['replicas'] if r.get('state')=='accepted']
                for source in sources:
                    spec=read(Path(source)/'spec.json');seed=spec['training']['seed']
                    destination=materialize_selected(source,root/'parents',spec,verify_native)
                    selected[role][seed]=dict(path=str(destination),manifest_sha256=file_sha256(destination/'selected_artifact.json'))
            seed_states={}
            for seed in (3416,3417,3418):
                if any(seed not in selected[role] for role in ('cfp','oct')):
                    seed_states[str(seed)]='waiting_this_seed_parents';continue
                if seed!=3416 and not pilot_accepted(root,name):
                    seed_states[str(seed)]='waiting_accepted_pilot_report';continue
                spec=dict(schema='radon_project_case_v1',model={'name':architecture},disease=disease,seed=seed,
                    parents={role:selected[role][seed] for role in selected},training=p['training'],arms=arms(disease,architecture),
                    bootstrap_iterations=10000,source_pins=config['source_pins'],test_access=False,
                    protocol_sha256=config['protocol']['sha256'],catalog_sha256=config['catalog']['sha256'])
                key=name+f'/seed{seed}'
                path=root/'specs'/(stable_hash(key)+'.json');path.parent.mkdir(exist_ok=True);immutable(path,spec)
                run=reserve(root,'radon_expanded_'+config['catalog']['sha256'][:16],key,spec,
                    source={'protocol':config['protocol'],'native_groups':name},refresh_summary=False)
                tasks.append(dict(id=key,spec=str(path),spec_sha256=file_sha256(path),run_dir=str(run),role='radon_project'))
                immutable(root/'bindings'/(stable_hash(key)+'.json'),dict(spec=str(path),run_dir=str(run),spec_sha256=file_sha256(path)))
                seed_states[str(seed)]='project_task_registered'
            states[name]=seed_states
    queue=dict(schema='radon_bridge_project_work_feed_v1',tasks=tasks,test_access=False)
    atomic_write_json(queue,root/'queue.json');atomic_write_json(coverage(),root/'coverage.json')
    from radon_bridge.analysis.project_rollup import summarize
    report=summarize(tasks,root/'report')
    return dict(tasks=len(tasks),accepted=report['accepted'],complete=report['complete'],groups=states,
        planned_training_positions=sum(len(arms(d,m)) for d in DISEASES for m in {r['model'] for r in read(config['catalog']['path'])['candidates']})*3,queue=str(root/'queue.json'))
