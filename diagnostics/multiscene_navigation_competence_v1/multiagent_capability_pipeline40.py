"""Original-state capability gates after full navigation; random obstacles remain locked."""
from common import *
from corrective_design40 import require_ready,prepare_capability,analyze_capability
from pilot_training40 import rollout_stage

def main():
    require_ready('A')
    for family in 'ABCDEF':
        nav=read(HERE/f'NAVIGATION_CAPABILITY_{family}.json') if (HERE/f'NAVIGATION_CAPABILITY_{family}.json').exists() else None
        if not nav or not nav['family_passed']:
            write(HERE/f'MULTIAGENT_CAPABILITY_{family}.json',dict(family=family,status='NOT_RUN',reason='Fewer than two qualified navigation instances',eligible=[],instances=[],family_passed=False));continue
        receipt=HERE/f'MULTIAGENT_CAPABILITY_{family}.receipt.json'
        if receipt.exists():
            assert all(sha(p)==h for p,h in read(receipt)['files'].items());continue
        stage='multiagent40_'+family
        if not (HERE/f'{stage}_CASES.json').exists():prepare_capability(family)
        rollout_stage(stage,'multiagent_worker40.py');analyze_capability(family)
    result={f:read(HERE/f'MULTIAGENT_CAPABILITY_{f}.json') for f in 'ABCDEF'}
    p=HERE/'AF_MULTIAGENT_CAPABILITY_GATES.json';write(p,dict(families=result,eligible_families=[f for f,v in result.items() if v['family_passed']],G_corrective_labels_locked=True))
    seal('AF_MULTIAGENT_CAPABILITY_GATES.receipt.json',[p],no_instance_replacement=True,G_not_opened=True,before_target_correction_evaluation_labels=True)
    write(HERE/'AF_MULTIAGENT_CAPABILITY_COMPLETE.json',dict(utc=utc(),eligible_families=[f for f,v in result.items() if v['family_passed']],next_stage='Probe and frozen selector study for unblocked families; report every blocked family',G_locked=True))

if __name__=='__main__':main()
