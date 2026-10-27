"""Verify the complete declared Cartesian label plan without reading outcomes."""
from common import *
import numpy as np


def main():
    scenes={s['name']:s for s in read(HERE/'SCENES.json')if s['family']!='extrapolation'}
    selected=read(HERE/'SELECTED_CONTROLLERS.json');menu={etahex(a['eta_a']):i for i,a in enumerate(actions())}
    assert len(menu)==49
    states={name:{s['index']:s for s in scene['states']}for name,scene in scenes.items()}
    controllers={name:{c['controller']:c for c in selected[name].values()if c is not None}for name in scenes}
    expected={(name,ctl,index,action)for name in scenes for ctl in controllers[name]for index in states[name]for action in range(49)}
    actual=set();checked_states=set();count=0
    for c in read(HERE/'main_CASES.json'):
        name=c['state']['scene'];index=c['state']['index'];ctl=c['controller_payload']['nominal']['controller']
        assert name in scenes and index in states[name]and ctl in controllers[name],('PLAN_DECLARATION_MISMATCH',c['rollout_uid'])
        a=menu[etahex(c['eta_a'])];cell=(name,ctl,index,a)
        assert cell not in actual,('DUPLICATE_LABEL_CELL',cell)
        actual.add(cell);count+=1
        key=(name,index)
        if key not in checked_states:
            declared=states[name][index]
            assert c['state']['N']==declared['N']
            for field in('positions','velocities','goals'):
                assert np.array_equal(np.asarray(c['state'][field]),np.asarray(declared[field])),('DECLARED_PHYSICAL_STATE_DIFF',key,field)
            checked_states.add(key)
        chosen=controllers[name][ctl];nominal=c['controller_payload']['nominal']
        for field in('type_code','parameters','checkpoint'):
            assert nominal.get(field)==chosen.get(field),('DECLARED_CONTROLLER_DIFF',name,ctl,field)
    assert actual==expected,('LABEL_PLAN_COVERAGE_DIFF',len(expected-actual),len(actual-expected))
    write(HERE/'MAIN_PLAN_COVERAGE_AUDIT.json',dict(utc=utc(),passed=True,outcomes_read=False,
        expected_cells=len(expected),actual_cells=count,scenes=len(scenes),physical_scene_states_checked=len(checked_states),
        each_scene_state_controller_action_exactly_once=True,all_declared_initial_physical_states_equal=True,
        all_selected_controller_parameters_equal=True,exact_float64_candidate_vectors=49,
        files={str(HERE/n):sha(HERE/n)for n in('SCENES.json','SELECTED_CONTROLLERS.json','main_CASES.json')}))
    print('Full label-plan Cartesian coverage and physical state/controller indexing passed:',count,flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(HERE/'MAIN_PLAN_AUDIT_ERROR.json',dict(utc=utc(),traceback=traceback.format_exc()))
        raise
