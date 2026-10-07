"""Executable regression: time alias fails before correction and separates after."""
import json
import numpy as np
from diagnostics.orthoflow3_closed_loop_contract_v1 import repair
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,phase_b as b

def run():
    witnesses=a.load(repair.OUT.parent/'waiting_probe/preregistration.json')['witnesses']
    by={s['uid']:s for s in a.load(repair.OUT.parent/'waiting_probe/states.json')}
    tests=[]
    for w in witnesses:
        x,y=[by[s]['physical'] for s in w['state_pair']]
        assert w['h_hashes'][0]==w['h_hashes'][1]
        assert x['timestep']!=y['timestep']
        for key in ('positions','velocities','goals'):assert np.array_equal(x[key],y[key])
        rx=repair.remaining({'structured_state':x});ry=repair.remaining({'structured_state':y})
        assert rx>ry and rx-ry>.99
        tests.append({'scenario':w['scenario'],'before_exact_alias':True,'after_time_separation':rx-ry})
    source=a.states();cohort=[]
    extra=a.load(a.OUT/'initial_expansion/states.json')
    for sc in ('four_way_intersection','ring_exchange'):
        rr=sorted([r for r in source if r['scenario']==sc],key=lambda r:a.digest('contract-static|'+r['state_uid']))[:6]
        rr+=sorted([r for r in extra if r['scenario']==sc],key=lambda r:a.digest('contract-static|'+r['state_uid']))[:6]
        for row in rr:
            p=a.decode(row['structured_state']);h=a.decode(row['conditioning'])['flat']
            assert np.isfinite(h).all() and np.isfinite(p['positions']).all()
            tr=repair.transformed(row)
            assert a.decode(tr['conditioning'])['flat'][-1]==repair.remaining(row)
            cohort.append({'state_uid':row['state_uid'],'scenario':sc,'split':row['split'],
                           'timestep':p['timestep'],'remaining_fraction':repair.remaining(row),
                           'physical_hash':a.digest(p),'input_hash':a.digest(a.decode(tr['conditioning']))})
    result={'time_witnesses':tests,'static_cohort':cohort,'passed':True,
            'future_seed_in_input':False,'eta_schedule':'fixed entire continuation',
            'labels_reencoding_only':'VALID_FOR_ORIGINAL_PHYSICAL_SNAPSHOT'}
    path=repair.OUT.parent/'contract_tests.json';path.write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
