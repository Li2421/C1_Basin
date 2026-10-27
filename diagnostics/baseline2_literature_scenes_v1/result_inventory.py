"""Resolve logical evaluation requests to existing exact raw evidence."""
from experiment import *
from copy import deepcopy

def logical(stage):
    own=HERE/stage/'raw';rows=[];sources=[]
    for c in read(HERE/f'{stage}_CASES.json'):
        rid=c['rollout_uid'];p=own/f'{rid}.json'
        if not p.exists():
            matches=[q for prefix in ('safety_','fixed_','learning_labels_','learning_eval_') for folder in HERE.glob(prefix+'*') if folder.is_dir() and (q:=folder/'raw'/f'{rid}.json').exists()]
            if not matches:raise FileNotFoundError((stage,rid))
            p=matches[0]
        r=deepcopy(read(p));assert r['rollout_uid']==rid
        assert all(r[k]==c[k] for k in ('state_uid','controller_uid','eta_uid','seed_key'))
        # Only the logical reporting name changes. Physical controller payload,
        # action schedule and registered identities are immutable.
        r['case']=c;rows.append(r);sources.append(dict(request=rid,source=str(p),sha256=sha(p)))
    return rows,sources

def qualification(scene):
    name=('capability_v2_' if scene=='lit_narrow_passage' else 'capability_')+scene
    return read(HERE/f'{name}_QUALIFICATION.json')

def export(scene):
    from migration_statistics import paired
    fixed=['safety','fixed_eta10','fixed_eta13','human_progress'];rows=[];sources=[]
    for stage in ('safety_'+scene,'fixed_'+scene):
        rr,ss=logical(stage);rows+=rr;sources+=ss
    full=paired(rows,qualification(scene),scene,fixed)
    result=dict(scene=scene,fixed_full_inventory=full,learned_evaluation=None)
    stage='learning_eval_'+scene
    if (HERE/f'{stage}_CASES.json').exists():
        rr,ss=logical(stage);rows+=rr;sources+=ss
        keys={(r['case']['state']['N'],r['case']['state']['index']) for r in rr}
        subset=[r for r in rows if (r['case']['state']['N'],r['case']['state']['index']) in keys]
        result['learned_evaluation']=paired(subset,qualification(scene),scene,fixed+['learned_scene'])
    write(HERE/f'MIGRATION_RESULTS_{scene}.json',result)
    write(HERE/f'MIGRATION_RESULT_SOURCES_{scene}.json',sources)
    seal(f'MIGRATION_RESULTS_{scene}.receipt.json',[HERE/f'MIGRATION_RESULTS_{scene}.json',HERE/f'MIGRATION_RESULT_SOURCES_{scene}.json',HERE/'result_inventory.py',HERE/'migration_statistics.py'],bootstrap_iterations=5000,two_seeds_same_state_cluster=True,unknown_lower_and_upper=True)
    print(scene,'complete',flush=True)

if __name__=='__main__':export(sys.argv[1])
