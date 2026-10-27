"""Exact semantic identities; reuse the already registered circle fallback."""
import resume_bootstrap
from experiment import *
from audit import V2
from registry_fix import case,register

def build(scene,variant='initial',override_gamma=None):
    receipt=read(HERE/'STAGE2_FIX_RESUME.receipt.json');assert receipt['scenes'][scene]['passed']
    assert read(HERE/'WRITE_PATH_AUDIT.json')['passed']
    if scene=='lit_circle_five':gamma=.3;variant='fallback'
    else:gamma=(.3 if scene=='lit_narrow_passage' else .2) if override_gamma is None else float(override_gamma)
    params=dict(gamma=gamma,circle_discrete=scene=='lit_circle_five',margin_fraction=.01,restoration='original_then_speed_only_then_safety_minimax')
    rows=[]
    for base in read(V2/f'optional_dynamics_{scene}_CASES.json'):
        scenario=dict(base['scenario_payload'],version='paper_dynamics_stage2_collision_fix_v1')
        payload=dict(base['controller_payload'],safety_fix=params,safety_source_sha256=sha(HERE/'stage2_safety.py'),runtime_sha256=sha(HERE/'runtime_fix.py'))
        c=case(base['state'],'paper_dynamics_stage2_collision_fix_v1',base['method'],scenario,payload,base['seed'],base['eta_a'],base['eta_b'],base['theta'],base['state_provenance'])
        c.update(safety_layer='distributed',execution_engine='paper_dynamics',rng_state_uid=base['rng_state_uid'],safety_fix=params,paired_v2_rollout_uid=base['rollout_uid'])
        rows.append(c)
    stage=f'{scene}_{variant}'
    if not (HERE/f'{stage}_REGISTRATION.receipt.json').exists():register(stage,rows)
    else:
        prior=read(HERE/f'{stage}_CASES.json');assert [c['rollout_uid'] for c in prior]==[c['rollout_uid'] for c in rows]
    first=read(HERE/f'FLOW_STATES_{scene}.json')[:4];keys={(s['N'],s['index']) for s in first}
    pilot=[c for c in rows if (c['state']['N'],c['state']['index']) in keys and c['seed']==0 and c['method'] in ('safety','fixed_eta13')]
    write(HERE/f'{stage}_PILOT_CASES_RESUME.json',pilot)
    print(stage,len(rows),'pilot',len(pilot),flush=True)
    return stage
if __name__=='__main__':build(*sys.argv[1:])
