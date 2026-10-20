"""New schedule identity/boundary tests plus two exact historical step replays."""
from common import *
from design import action_case
from candidates import catalog,draw,encode,TIMES
def main():
    base=read(D/'crosstype1b_generator_critic_pipeline_v1/BASE_PROTOCOL.json');states=read(D/'crosstype1b_generator_critic_pipeline_v1/TRAINING_STATES.json');r=read(D/'crosstype1b_generator_critic_pipeline_v1/CANDIDATES.json')['states'][0];eta=np.asarray([*r['proposals'],[.8,.1,.1],[.9,.2,.2],[1.,.3,.3]])
    aa=catalog(eta);assert len(aa)==320 and sum(a['unique_representative'] for a in aa)==288
    for a in aa:
        assert len(encode(a))==10
        for t in (a['switch_step']-1,a['switch_step']):
            actual=a['eta_a'] if t<a['switch_step'] else a['eta_b'];expected=eta[a['early_index'] if t<a['switch_step'] else a['late_index']];assert np.array_equal(actual,expected)
    ids=draw(base['controller_order'][0],states[0]['state_uid']);assert ids==draw(base['controller_order'][0],states[0]['state_uid']) and len(set(ids))==26
    for k in range(5):assert sum(i%5==k for i in ids[8:])>=3
    name=base['controller_order'][0];c,profile=action_case(base,name,states[0],aa[3],0,0,3);assert profile['controller_uid']==base['profiles'][name]['controller_uid']
    from generated_execution import execute
    old=D/'crosstype1b_generator_critic_pipeline_v1';cases=read(old/'new_evaluation_CASES.json');p=read(old/'new_evaluation_protocol.json');checks=[]
    for kind in ('Flow','diffusion'):
        c=next(c for c in cases if c['family']==kind and c['candidate_index']==11 and c['seed']==0);ref=read(old/'new_evaluation/raw'/f"{c['rollout_uid']}.json");result=execute(c,p,{},'schedule_tests')
        with np.load(ref['trace_path'],allow_pickle=False) as a,np.load(result['trace_path'],allow_pickle=False) as b:equal=a['positions'].tobytes()==b['positions'].tobytes() and a['positions'].shape==b['positions'].shape
        checks.append(dict(kind=kind,positions_bytes=equal,termination=ref['outcome']==result['outcome'],steps=ref['episode_length']==result['episode_length'],first_action_bytes=np.asarray(ref['first_action'],np.float64).tobytes()==np.asarray(result['first_action'],np.float64).tobytes()))
    passed=all(all(v for k,v in row.items() if k!='kind') for row in checks)
    write(HERE/'GENERATED_ACTION_TESTS.json',dict(utc=utc(),passed=passed,identity_retains_fixed_cache=True,boundary_all320_cases=True,balanced_18_draw_reproducible=True,step_replays=checks,new_labels=0));seal('GENERATED_ACTION_TESTS.receipt.json',[HERE/'GENERATED_ACTION_TESTS.json',HERE/'test_schedule.py',HERE/'design.py'])
    assert passed
if __name__=='__main__':main()
