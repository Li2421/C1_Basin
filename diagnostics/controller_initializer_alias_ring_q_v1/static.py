"""Verify the precise same alias with the frozen neural Flow reference."""
import argparse, json
import numpy as np
from .study import HERE,BASE,ROOT,SCENE,FAMILIES,read,write,freeze,sha


def prepare():
    from shared_rollout_db.src.rollout_db import connect
    p=read(BASE/'protocol.json')
    states=sorted([s for s in p['states'] if s['scenario']==SCENE and s['study_split']=='train'],key=lambda s:s['study_index'])[:8]
    freeze(HERE/'static_protocol.json',dict(states=states,families=FAMILIES,eta=p['eta'],eta_indices=[0,5,10,15],
        alpha=2.,field_key=2026100908,action_key=2026101204,action_folds=[0,1,2,3],
        scope='First8 old natural TRAINstates x4ALLuncoupled Flow-reference operators. Every old input/action reused, first case replay. No Env.step. Same construction as goal audit, only reference class changes.',
        history='No old initializer-alias profiles; old exact6+init8 tests and Four goal Q alias already completed. Four lacks selection headroom; not training there.',
        code_sha256=sha(__file__)))
    with connect(True) as db:
        records=[dict(controller_uid=r['controller_uid'],config=json.loads(r['config_json'])) for r in db.execute('SELECT controller_uid,config_json FROM controller_config')]
    freeze(HERE/'static_cache_audit.json',dict(profiles_scanned=len(records),inventory=records,
        new_rollouts=0,source_cache=str(BASE/'features'),evidence_gap='No altered Flow-wrapper initializer label or fingerprint; original compatible cache remains valid.'))


def worker(index):
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime,controller_context
    from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
    from diagnostics.controller_q_diversity_v1.features import field_context
    from diagnostics.controller_q_diversity_v1.train import compact_field
    p=read(HERE/'static_protocol.json');assert p['code_sha256']==sha(__file__)
    base=native_runtime(SCENE);name=FAMILIES[index]
    rt=FlowEndpointMatchedField(base,name.removeprefix('flow_'))
    alt=FlowEndpointMatchedField(base,name.removeprefix('flow_'))
    z=np.asarray(jax.random.normal(jax.random.PRNGKey(p['field_key']),(1,base.agent.config['joint_act_dim']))).reshape(-1,2)
    alt.scale=2*base.scale;alt.mean=base.mean+(base.scale-alt.scale)*z
    def forbidden(*a,**k):raise AssertionError('No physical simulation in static fingerprint audit')
    class Interface:
        def __init__(self,rt):self.rt=rt
        def __getattr__(self,k):return getattr(self.rt,k)
        def make_env(self,state):
            e=self.rt.make_env(state);e.step=forbidden;return e
    rows=[]
    for si,s in enumerate(p['states']):
        c=dict(np.load(BASE/'features'/s['state_uid']/(name+'.npz')))
        nominal=controller_context(Interface(alt),s);field=field_context(Interface(alt),s)
        current=np.asarray(compact_field(np.concatenate((nominal,field),-1)))
        if si==0:
            np.testing.assert_allclose(field_context(Interface(rt),s),c['field'],atol=1e-6,rtol=0)
            np.testing.assert_array_equal(controller_context(Interface(rt),s),c['nominal'])
        env=rt.make_env(s);env.step=forbidden;actions=np.full_like(c['current_raw'],np.nan);ok=np.zeros_like(c['current_ok']);errors=[]
        for j in p['eta_indices']:
            for k in p['action_folds']:
                key=jax.random.fold_in(jax.random.PRNGKey(p['action_key']),k)
                if si==0 and j==0 and k==0:np.testing.assert_allclose(rt.action(env,key,np.asarray(p['eta'][j]),'field')[0],c['current_raw'][j,k],atol=2e-6,rtol=0)
                try:actions[j,k]=alt.action(env,key,np.asarray(p['eta'][j]),'field')[0];ok[j,k]=True
                except Exception as ex:errors.append(dict(eta=j,seed=k,type=type(ex).__name__,message=str(ex)[:300]))
        dest=HERE/'static_raw'/s['state_uid']/(name+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,original_compact=c['compact'],alternate_compact=current,
            original_action=c['current_raw'],alternate_action=actions,original_ok=c['current_ok'],alternate_ok=ok)
        ii=np.asarray(p['eta_indices']);valid=c['current_ok'][ii]&ok[ii]
        difference=np.max(abs(actions[ii]-c['current_raw'][ii]),axis=(-2,-1))/base.config.max_speed
        rows.append(dict(state_uid=s['state_uid'],controller=name,compact_max_error=float(np.max(abs(current-c['compact']))),
            action_cases=int(valid.sum()),changed_action_cases=int((difference[valid]>1e-5).sum()),max_action_difference=float(difference[valid].max()),errors=errors))
    write(HERE/f'static{index}.json',rows)


def review():
    rows=sum([read(HERE/f'static{i}.json') for i in range(4)],[])
    result=dict(rows=rows,static_input_alias_verified=all(r['compact_max_error']<1e-5 for r in rows),
        original_actions_reused=sum(r['action_cases'] for r in rows),changed_actions=sum(r['changed_action_cases'] for r in rows),
        new_rollout=0,new_fit=0)
    write(HERE/'static_results.json',result);print({k:v for k,v in result.items() if k!='rows'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','review']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    worker(a.index) if a.action=='worker' else globals()[a.action]()
