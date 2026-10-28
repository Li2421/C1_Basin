"""Replay initial actions from saved evidence; never advance the simulator."""
from core import *
from acceleration import enable,accelerated_case
from runtime import fields_for,ROUTE_EVALUATE
from routes import guidance,route_geometry
from shared_control.basis_dictionary import RoleState,add_target
import hashlib

OLD=HERE.parent/'scene_types_basis_dimension_v1'

def first_action(c):
    env=RN.SwapEnv(c['state'],RN.LiteratureConfig(**c['scenario_payload']['config']));cfg=env.config
    actor=RN.Nominal(c['controller_payload']['nominal']);true_goals=env.goals.copy()
    env._route_enabled=bool(c['scenario_payload']['route_guidance'])
    geom=route_geometry(env.instance.geometry)
    root=RN.jax.random.fold_in(RN.jax.random.fold_in(RN.jax.random.PRNGKey(2026102004),int(hashlib.sha256(c['rng_state_uid'].encode()).hexdigest()[:8],16)),c['seed'])
    guides,remaining,_=guidance(geom,env.positions,true_goals,env._route_enabled)
    env.goals=guides
    ref=RN.bound(actor.action(env,RN.jax.random.fold_in(root,0)),cfg.max_speed).astype(np.float64)
    ref[np.linalg.norm(true_goals-env.positions,axis=-1)<=cfg.goal_tolerance]=0.
    env.goals=true_goals
    filters=[RN.SwapSafety(env.snapshot(),i) for i in range(env.num_agents)]
    v=np.concatenate([f.project(ref[i:i+1])[0] for i,f in enumerate(filters)])
    ff=fields_for(env,v,guides,remaining);sp=c['spec'];eta=np.asarray(sp['eta']);co=sp.get('coefficients',{})
    target=v.copy()
    for i in range(env.num_agents):target[i]+=sum(float(x)*field for x,field in zip(eta,ff[i]))
    if co:
        cols,_=ROUTE_EVALUATE(env.positions,guides,v,ref,ff,filters,env.walls,cfg,RoleState(),0,names=co,route_remaining=remaining)
        target=add_target(v,ff,cols,eta,co)
    action=np.concatenate([f.project(target[i:i+1])[0] for i,f in enumerate(filters)])
    return dict(nominal=ref,safe=v,actions=action,route_guides=guides,route_remaining=remaining)

def main():
    enable()
    audit=read(HERE/'HISTORY_AUDIT.json')
    scenes={e['type_id']:e['scene'] for e in read(OLD/'SCENES_FROZEN.json')['entries'] if e.get('scene')}
    ctls={c['type_code']:c for c in controllers()}
    uids={u for p in audit['proposals'] for q in p['untried_successful_search_points'] for u in q['source_uids']}
    held=read(OLD/'HOLDOUT_LABELS.json')['0']
    for p in audit['proposals']:
        matches=[x for x in held if (x['type_id'],x['controller'],x['state_index'])==(p['type_id'],p['controller'],p['state_index'])]
        if matches:uids.add(matches[0]['uid'])
    results=[];skipped=[]
    for uid in sorted(uids):
        r=read(OLD/'raw'/f'{uid}.json')
        with np.load(r['trace'],allow_pickle=False) as saved:
            if not len(saved['actions']):
                skipped.append(dict(uid=uid,reason='Historical unknown before first recorded action; retained, no simulator replay.'));continue
            c=accelerated_case(scenes[r['type_id']],ctls[r['controller']],r['state_index'],r['spec'],purpose='action_replay_only')
            assert c['state_uid']==r['state_uid'] and c['scenario_uid']==r['scenario_uid']
            outputs=first_action(c)
            checks={k:(v.dtype==saved[k][0].dtype and v.shape==saved[k][0].shape and v.tobytes()==saved[k][0].tobytes()) for k,v in outputs.items()}
            assert all(checks.values()),(uid,checks)
            results.append(dict(uid=uid,controller=r['controller'],type_id=r['type_id'],checks=checks))
    dump(HERE/'TRANSPORT_REGRESSION.json',dict(passed=True,actual_env_steps=0,matched_initial_actions=len(results),skipped_historical_unknowns=skipped,results=results,reused_full_episode_certificate=str(OLD/'ACCELERATION_CERTIFICATE.json'),scope='Byte-identical numerical modules; metadata-only namespace/paths/cache changes. Initial action replay is not a new whole-episode equivalence certificate.'))
    print('PASS initial actions',len(results),'skipped historical initial unknowns',len(skipped),flush=True)

if __name__=='__main__':main()
