from common import *
from generated_execution import materialize


def main():
    previous=read(D/'crosstype1b_generator_critic_pipeline_v1/CANDIDATES.json')
    current=read(HERE/'CANDIDATES.json');p=read(D/'crosstype1b_generator_critic_pipeline_v1/BASE_PROTOCOL.json');before=clean(p)
    for i,row in enumerate(current['states']):
        assert len(row['actions'])==16
        if i<48:
            assert row['actions'][:13]==previous['states'][i]['actions']
            assert row['proposals']==previous['states'][i]['proposals']
        for j,a in enumerate(row['actions']):
            if j<10:assert a['eta_a']==row['proposals'][j//2]
            if a['switch_mode']=='progress':
                req=dict(arm=a['name'],**{k:a[k] for k in ('eta_a','eta_b','switch_mode')})
                c,pp=materialize(req,p);assert c['arm']=='Sprog' and pp['eta'][13]==a['eta_a'] and pp['eta'][10]==a['eta_b']
            if j>=13:assert a['eta_a']==a['eta_b']==[0.,0.,0.] and a['rhr_rule']['speed']>0
    assert clean(p)==before
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:
        assert z['h'].shape==(72,100) and z['c'].shape==(72,33) and z['action'].shape==(72,16,16)
        assert np.isfinite(z['action']).all()
    write(HERE/'GENERATED_ACTION_TESTS.json',dict(utc=utc(),passed=True,training_old13_actions_bitwise_reused=True,all_16_actions=72*16,original_eta_bank_unchanged=True,new_rollouts=0))


if __name__=='__main__':main()
