"""New execution identity, official append-only registry and preflight."""
from experiment import *
from audit import V2
from registry_fix import case,register

def main(scene,variant):
    gate=read(HERE/'STAGE2_FIX.receipt.json')
    assert gate['scenes'][scene]['passed'],('unit_gate_failed',scene)
    assert read(HERE/'REPLAY_GATE.json')['passed'],'historical_replay_gate_failed'
    path=V2/f'optional_dynamics_{scene}_CASES.json'
    with (HERE/'EVIDENCE_INPUTS.jsonl').open('a') as f:f.write(canonical(dict(path=str(path),sha256=sha(path)))+'\n')
    old=read(path);rows=[]
    params=dict(gamma=gate['scenes'][scene].get('gamma'),circle_discrete=variant=='fallback',
                margin_fraction=.01,restoration='original_then_speed_only_then_safety_minimax')
    for base in old:
        scenario=dict(base['scenario_payload'],version='paper_dynamics_stage2_collision_fix_v1')
        payload=dict(base['controller_payload'],safety_fix=params,
                     safety_source_sha256=sha(HERE/'stage2_safety.py'),runtime_sha256=sha(HERE/'runtime_fix.py'))
        c=case(base['state'],'paper_dynamics_stage2_collision_fix_v1',base['method'],scenario,payload,
               base['seed'],base['eta_a'],base['eta_b'],base['theta'],base['state_provenance'])
        c.update(safety_layer='distributed',execution_engine='paper_dynamics',rng_state_uid=base['rng_state_uid'],
                 safety_fix=params,paired_v2_rollout_uid=base['rollout_uid'])
        rows.append(c)
    stage=scene+'_'+variant
    register(stage,rows)
    first=read(HERE/f'FLOW_STATES_{scene}.json')[:4]
    keys={(s['N'],s['index']) for s in first}
    planned=read(HERE/f'{stage}_MISSING_CASES.json')
    pilot=[c for c in planned if (c['state']['N'],c['state']['index']) in keys and c['seed']==0 and c['method'] in ('safety','fixed_eta13')]
    write(HERE/f'{stage}_PILOT_CASES.json',pilot)
    print('pilot',len(pilot),'full',len(rows),flush=True)
if __name__=='__main__':main(*sys.argv[1:])
