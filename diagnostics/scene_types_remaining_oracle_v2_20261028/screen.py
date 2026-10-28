"""Small posthoc transfer screen, with unchanged numerical execution semantics."""
import os
os.environ['STB_COORDINATOR']='1'
from acceleration import accelerated_case
from core import *
from dispatch import execute

OLD=HERE.parent/'scene_types_basis_dimension_v1'

def planned():
    audit=read(HERE/'HISTORY_AUDIT.json')
    scenes={x['type_id']:x['scene'] for x in read(OLD/'SCENES_FROZEN.json')['entries'] if x.get('scene')}
    ctls={x['type_code']:x for x in controllers()}
    cases=[]
    for item in audit['proposals']:
        for point in item['untried_successful_search_points']:
            c=accelerated_case(scenes[item['type_id']],ctls[item['controller']],item['state_index'],point['spec'],purpose='posthoc_development_success_transfer')
            c['transfer_source_uids']=point['source_uids']
            c['transfer_source_search_states']=point['source_search_states']
            assert all(i in range(4) for i in c['transfer_source_search_states'])
            cases.append(c)
    assert len(cases)==len({c['rollout_uid'] for c in cases})==56
    return cases

def main():
    assert read(HERE/'TRANSPORT_REGRESSION.json')['passed']
    cases=planned()
    # Register/preflight the entire immutable request set before starting.
    from runner import register
    register(cases,'TRANSFER56')
    smoke=[next(c for c in cases if c['controller_payload']['nominal']['type_code']==k) for k in ('centralized_sampling','gradient_mpc')]
    rows=execute(smoke,'TRANSFER_SMOKE')
    assert len(rows)==2
    execute(cases,'TRANSFER56')

if __name__=='__main__':main()
