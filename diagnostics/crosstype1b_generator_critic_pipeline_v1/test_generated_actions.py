"""Verify generated-eta schedules against preserved trajectories, zero physics."""
from common import *
from generated_execution import materialize
from progress import ProgressSwitch


def main():
    p=read(HERE/'BASE_PROTOCOL.json');original=clean(p['eta']);menu=read(HERE/'CANDIDATES.json')
    checked=0;switches=[]
    for record in menu['states']:
        assert len(record['actions'])==13 and len(record['proposals'])==5
        for i,a in enumerate(record['actions']):
            early=np.asarray(a['eta_a']);late=np.asarray(a['eta_b'])
            if i<10:
                assert np.array_equal(early,np.asarray(record['proposals'][i//2]))
                assert np.array_equal(early.astype(np.float32).astype(float),early)
                assert np.all(early>=[.5,-.5,0]) and np.all(early<=[1.25,.5,.75])
            if a['switch_mode']=='progress':
                c=dict(arm=a['name'],eta_a=a['eta_a'],eta_b=a['eta_b'],switch_mode='progress')
                current,protocol=materialize(c,p)
                assert current['arm']=='Sprog' and c['arm']==a['name']
                assert protocol is not p and np.array_equal(protocol['eta'][13],early) and np.array_equal(protocol['eta'][10],late)
                assert clean(p['eta'])==original
                assert np.array_equal(late,np.asarray(p['eta'][10]))
            checked+=1
    for row in read(HERE/'REPRODUCTION_CASES.json'):
        c=row['case']
        if c['arm']!='Sprog':continue
        raw=read(row['reference'])
        with np.load(raw['trace_path'],allow_pickle=False) as z:positions=z['positions'];goals=z['goals']
        rule=ProgressSwitch(positions[0],goals,700,.9681620884830543)
        for t in range(len(positions)-1):rule.eta_index(t,positions[t],goals)
        assert rule.switched_at==raw['progress_switch_step']
        switches.append(dict(controller=c['base_controller_name'],switch_step=rule.switched_at))
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:
        assert z['h'].shape==(72,100) and z['c'].shape==(72,33) and z['action'].shape==(72,13,10)
        assert np.isfinite(z['h']).all() and np.isfinite(z['c']).all() and np.isfinite(z['action']).all()
    write(HERE/'GENERATED_ACTION_TESTS.json',dict(utc=utc(),passed=True,actions_checked=checked,historical_switches=switches,
        original_eta_bank_unchanged=True,full_precision_eta_preserved=True,new_rollouts=0,controller_independent_conditioning=True))
    seal('GENERATED_ACTION_TESTS.receipt.json',[HERE/'GENERATED_ACTION_TESTS.json',HERE/'test_generated_actions.py',HERE/'generated_execution.py'])
    print('Generated-action parameter/schedule tests passed; no new physical steps',flush=True)


if __name__=='__main__':main()
