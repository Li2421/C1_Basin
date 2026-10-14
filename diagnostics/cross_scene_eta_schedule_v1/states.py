from core import *
from shared_rollout_db.src.rollout_db import uid,canonical,connect
from collections import Counter

def pack(sc,i,physical,metadata):
    old=read(E8/'protocol.json');sid=old['environments'][sc]['scenario_uid'];alias=f'round13_schedule_fresh_{sc}_{i}';content=uid('content',dict(physical=physical,reset='empty monitor true-t0'));return dict(uid=alias,state_uid=uid('state',dict(scenario=sid,content=content)),content_hash=content,scenario=sc,scenario_uid=sid,rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),physical=physical,family=alias,study_split='prospective_round13',study_index=i,metadata=dict(metadata,split='prospective_round13',outcome_blind=True))
def worker(part):
    prereg();old=read(E8/'protocol.json')
    if part<3:
        import jax
        from diagnostics.controller_q_diversity_v1.runtime import native_runtime
        native_runtime('double_bottleneck');jax.config.update('jax_enable_x64',False)
        from diagnostics.double_bottleneck_initial_state_coverage.tools.generate_pools import _sample_spec,_solve_spec,_spec_json
        from double_bottleneck.scenario import INITIAL_REGIMES,build_layout
        from double_bottleneck.environment import Config
        assert clean(Config().to_dict())==old['environments']['double_bottleneck']['config']['config'];regime=INITIAL_REGIMES[part];seed=int.from_bytes(hashlib.sha256(f'round13_db_global_v1|2026234000|validation|{regime}'.encode()).digest()[:8],'little');rng=np.random.default_rng(seed);states=[];rejected=[];k=0;goal=[14,13,13][part]
        while len(states)<goal:
            spec=_sample_spec('validation',regime,k,rng);result=_solve_spec(('validation','val',spec))
            if result['accepted']:
                physical=dict(positions=spec.positions.tolist(),velocities=spec.initial_velocities.tolist(),goals=build_layout(Config()).goals.tolist(),timestep=0);states.append(pack('double_bottleneck',sum([14,13,13][:part])+len(states),physical,dict(regime=regime,draw_index=k,rng_seed=seed,original_sampler_spec=_spec_json(spec))))
            else:rejected.append(dict(candidate=k,reason=result['reason']))
            k+=1;assert k<=4096;print(json.dumps(dict(part=part,candidates=k,accepted=len(states),needed=goal)),flush=True)
        freeze(HERE/f'states_part{part}.json',dict(states=states,audit=dict(regime=regime,rng_seed=seed,draws=k,rejections=rejected,accepted=len(states),auxiliary_expert_mode_count=8,acceptance_source='original _solve_spec, no target FF filtering')))
    else:
        sc=SCENES[[0,2,3][part-3]];cfg=old['environments'][sc]['config']['config'];states=[]
        if sc=='toy_give_way':
            for i in range(40):
                seed=2026231000+i;rng=np.random.default_rng(seed);x=rng.uniform(.55,1.05,2);y=rng.uniform(-.025,.025,2);pos=np.array([[-x[0],y[0]],[x[1],y[1]]],np.float32).astype(float);states.append(pack(sc,i,dict(positions=pos.tolist(),velocities=[[0.,0.],[0.,0.]],goals=[[1.09,0.],[-1.09,0.]],timestep=0),dict(draw_seed=seed)))
        elif sc=='four_way_intersection':
            from four_way_intersection.environment import Config
            from four_way_intersection.scenario import sample_initial_state,build_layout
            config=Config(**cfg)
            for i in range(40):
                pos,vel,metadata=sample_initial_state(config,'development',i,base_seed=2026233000);states.append(pack(sc,i,dict(positions=pos.tolist(),velocities=vel.tolist(),goals=build_layout(config).goals.tolist(),timestep=0),dict(metadata,draw_seed=2036233000+i)))
        else:
            from ring_exchange.environment import LocalFrameConfig,sample_instance
            config=LocalFrameConfig(**cfg)
            for i in range(40):
                seed=2026232000+i;ins=sample_instance(seed,'development',config);states.append(pack(sc,i,dict(positions=ins.positions.tolist(),velocities=ins.velocities.tolist(),goals=ins.goals.tolist(),timestep=0),dict(draw_seed=seed)))
        freeze(HERE/f'states_part{part}.json',dict(states=states,audit=dict(scene=sc,accepted=40,no_target_outcome_filter=True)))
def signature(v):
    if not isinstance(v,dict):return None
    values=[]
    for ks in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
        val=next((v[k] for k in ks if k in v),None)
        if val is None:return None
        a=np.asarray(val,np.float64)
        if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all():return None
        a=a.copy();a[a==0]=0;values.append((a.shape,a.tobytes().hex()))
    return hashlib.sha256(canonical(dict(values=values,timestep=float(v.get('timestep',v.get('step',0))))).encode()).hexdigest()
def collect():
    prereg();parts=[read(HERE/f'states_part{i}.json') for i in range(6)];states=sorted([s for p in parts for s in p['states']],key=lambda s:(SCENES.index(s['scenario']),s['study_index']));assert len(states)==160 and all(sum(s['scenario']==sc for s in states)==40 for sc in SCENES);sig={signature(s['physical']):s['state_uid'] for s in states};assert None not in sig and len(sig)==160;ids={s['state_uid'] for s in states};contents={s['content_hash'] for s in states};collisions=[];unresolved=Counter();recovered=0
    with connect(True) as db:old=db.execute('SELECT state_uid,content_hash,physical_state_json,identity_quality FROM state').fetchall()
    for r in old:
        if r['state_uid'] in ids or r['content_hash'] in contents:collisions.append(dict(source='DBUID/content',state_uid=r['state_uid']))
        try:v=json.loads(r['physical_state_json']) if r['physical_state_json'] else None
        except ValueError:v=None
        h=signature(v)
        if h is None:unresolved[r['identity_quality']]+=1
        else:
            recovered+=1
            if h in sig:collisions.append(dict(source='DBphysical',state_uid=r['state_uid']))
    checked=[]
    for path in ROOT.glob('diagnostics/**/protocol.json'):
        if path.is_relative_to(HERE):continue
        q=read(path)
        if not isinstance(q,dict):continue
        rows=q.get('states',[]);rows=list(rows.values()) if isinstance(rows,dict) else rows
        if not isinstance(rows,list):continue
        for s in rows:
            if isinstance(s,dict) and signature(s.get('physical')) in sig:collisions.append(dict(source=str(path),state_uid=s.get('state_uid',s.get('uid'))))
        checked.append(dict(path=str(path),sha256=sha(path)))
    freeze(HERE/'FRESH_STATES.json',states);freeze(HERE/'STATE_DEDUP_AUDIT.json',dict(utc=utc(),passed=not collisions,fresh=160,all_DB_rows=len(old),recoverable=recovered,legacy_unresolved=unresolved,collisions=collisions,historical_protocols=checked,generation_parts=[p['audit'] for p in parts],no_target_labels_read_or_generated=True));assert not collisions,'Overlap: stop, no adaptive redraw';print('Fresh160 dedup PASS')
if __name__=='__main__':collect() if sys.argv[1]=='collect' else worker(int(sys.argv[1]))
