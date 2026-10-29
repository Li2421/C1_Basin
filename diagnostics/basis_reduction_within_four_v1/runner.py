"""Official DB registration, preflight, one-CPU workers and immutable traces."""
from core import *
import os,time,subprocess

def sources(spec=None):
    paths=[HERE/n for n in ('runtime.py','routes.py','core.py','runner.py','guard.py','budget.py','recording_fix.py','admission.py','admission_legacy.py','passages.py')]
    paths += [ROOT/'shared_control/basis_dictionary.py',ROOT/'shared_control/hard_projection.py',ROOT/'shared_control/basis_families.py',ROOT/'shared_control/correction_b5.py']
    
    return {str(p):sha(p)for p in paths}|{'private_harness':digest(read(HERE/'BACKEND_BUILD.json')), 'passages':sha(HERE/'PASSAGES.json')}

def case(scene,controller,index,spec=None,solo=None,force_route=None,purpose='main'):
    state=json.loads(json.dumps(scene['states'][index]));spec=spec or dict(eta=[0.,0.,0.],coefficients={})
    if solo is not None:
        state.update(N=1,positions=[state['positions'][solo]],velocities=[state['velocities'][solo]],goals=[state['goals'][solo]],solo_parent_agent=solo)
    config=dict(scene['config'],num_agents=state['N'])
    enabled=scene['route_guidance']if force_route is None else force_route
    sc=dict(version='scene_types_oracle_coverage_v3',config=config,bottleneck_config=scene['bottleneck_config'],walls=scene['walls'],factor=scene['type_id'],route_guidance=bool(enabled),plant='frozen_swap_step',scene_definition_sha256=scene['definition_sha256'])
    scid=db.uid('scenario',sc);physical={k:state[k]for k in ('positions','velocities','goals')}
    sid=db.uid('state',dict(scenario=scid,physical=physical))
    ctl=dict(nominal=controller,dictionary=spec,execution_sources=sources(spec),safety_layer='distributed',precision='jax_x64_false',correction_semantics='original_dictionary_directions_route_remaining_input_v1',rng='2026102004/matched_physical_state/seed/t')
    cid=db.uid('ctl',dict(scenario=scid,payload=ctl));eid=db.eta_identity(spec['eta'])[0];sk=db.canonical(dict(future_index=0))
    return dict(state=state,scenario_payload=sc,controller_payload=ctl,scenario_uid=scid,state_uid=sid,controller_uid=cid,eta_uid=eid,seed_key=sk,
        rollout_uid=db.uid('roll',dict(state=sid,eta=eid,controller=cid,seed=sk)),seed=0,eta_a=spec['eta'],eta_b=spec['eta'],theta=None,spec=spec,
        scene=scene['name'],type_id=scene['type_id'],state_index=index,solo=solo,purpose=purpose,budget_group=os.environ.get('OWA_BUDGET_GROUP','original'),
        rng_state_uid=db.uid('rng_physical',physical),state_provenance=dict(experiment=EXP,scene_definition_sha256=scene['definition_sha256'],state_index=index,solo_agent=solo,generation_seed=2026102801,purpose=purpose))

def register(cases,stage):
    assert os.environ.get('BRD_COORDINATOR')=='1'
    cases=list({c['rollout_uid']:c for c in cases}.values())
    with db.connect()as con,db.transaction(con):
        con.execute('INSERT INTO experiment(experiment_uid,name,path,metadata_json)VALUES(?,?,?,?) ON CONFLICT(experiment_uid)DO NOTHING',(EXP,EXP,str(HERE),db.canonical(dict(scope='basis-reduction-within-four-v1',no_training=True))))
        for c in cases:
            physical={k:c['state'][k]for k in ('positions','velocities','goals')};prov=db.canonical(c['state_provenance'])
            records={
                'scenario':dict(scenario_uid=c['scenario_uid'],name=c['scene'],code_config_fingerprint=c['scenario_uid'],metadata_json=db.canonical(c['scenario_payload'])),
                'state':dict(state_uid=c['state_uid'],scenario_uid=c['scenario_uid'],source_group=f"{c['scene']}_state{c['state_index']}_solo{c['solo']}",content_hash=digest(physical),physical_state_json=db.canonical(physical),goals_geometry_json=db.canonical(c['scenario_payload']),provenance_json=prov,identity_quality='CONTENT_EXACT'),
                'controller_config':dict(controller_uid=c['controller_uid'],scenario_uid=c['scenario_uid'],config_json=db.canonical(c['controller_payload']),compatibility_quality='EXACT_PROFILE')}
            for table,values in records.items():
                key=next(iter(values));row=con.execute(f'SELECT * FROM {table} WHERE {key}=?',(values[key],)).fetchone()
                if row:
                    # Same physical state may legitimately participate in multiple stages.
                    compare={k:v for k,v in values.items()if k!='provenance_json'}
                    assert all(row[k]==v for k,v in compare.items()),('DB_IDENTITY_CONFLICT',table,values[key])
                    if table=='state':assert json.loads(row['provenance_json']).get('generation_seed')==2026102801
                else:
                    con.execute(f'INSERT INTO {table}('+','.join(values)+')VALUES('+','.join('?'for _ in values)+')',list(values.values()))
                saved=con.execute(f'SELECT * FROM {table} WHERE {key}=?',(values[key],)).fetchone()
                assert saved
                if table=='state':
                    assert saved['physical_state_json']==db.canonical(physical)
                    assert saved['goals_geometry_json']==db.canonical(c['scenario_payload'])
                    assert json.loads(saved['provenance_json']).get('generation_seed')==2026102801
            eid,e,hx=db.eta_identity(c['eta_a']);assert eid==c['eta_uid']
            con.execute('INSERT INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex)VALUES(?,?,?,?,?)ON CONFLICT(eta_uid)DO NOTHING',(eid,*e,hx))
    path=HERE/f'{stage}_planned.json';existing=read(path)if path.exists()else None
    plan=dict(requests=[{k:c[k]for k in ('state_uid','eta_uid','controller_uid','seed_key')}for c in cases],exact_configs=cases)
    if existing:assert digest(existing)==digest(plan),('IMMUTABLE_STAGE_PLAN',stage)
    else:dump(path,plan)
    pf=planner.preflight(path);dump(HERE/f'{stage}_preflight.json',pf)
    assert pf['summary']['ambiguous']==0,('DB_AMBIGUOUS',pf['summary'])
    missing={(r['state_uid'],r['eta_uid'],r['controller_uid'],s)for r in pf['details']for s in r['missing_seeds']}
    # Numerical unknowns are retained even though official reusable-success lookup excludes them.
    done={}
    with db.connect(True)as con:
        for c in cases:
            uid=c['rollout_uid'];raw=HERE/'raw'/f'{uid}.json'
            if raw.exists():done[uid]=read(raw)
            row=con.execute('SELECT * FROM rollout WHERE rollout_uid=?',(uid,)).fetchone()
            if row:
                assert not row['conflict_quarantined']
                if uid not in done:
                    source=con.execute('SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf ON rs.source_uid=sf.source_uid WHERE rs.rollout_uid=? LIMIT 1',(uid,)).fetchone()
                    if source:
                        with Path(source['path']).open()as f:
                            done[uid]=next(json.loads(line)for i,line in enumerate(f,1)if i==source['source_line'])
                    else:raise RuntimeError('Existing exact result has no readable raw source: '+uid)
                assert done[uid]['success']==row['success']
            elif uid not in done:
                assert tuple(c[k]for k in ('state_uid','eta_uid','controller_uid','seed_key'))in missing
    return cases,[c for c in cases if c['rollout_uid']not in done],done

def work(c):
    if (HERE/'PROTECTION_VIOLATION.jsonl').exists():raise RuntimeError('HARD_STOP protection event')
    from runtime import run
    t=time.monotonic();result,arrays=run(c)
    folder=HERE/'traces';folder.mkdir(exist_ok=True);path=folder/f"{c['rollout_uid']}.npz"
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)
    r=dict(result,**{k:c[k]for k in ('scenario_uid','state_uid','controller_uid','eta_uid','seed_key','rollout_uid','scene','type_id','state_index','seed','spec','purpose','solo')},
        controller=c['controller_payload']['nominal']['type_code'],eta=c['eta_a'],future_index=c['seed'],cache_identity_schema='registered_exact_uid_v1',
        initial_positions=c['state']['positions'],initial_velocities=c['state']['velocities'],goals=c['state']['goals'],deadlock=0,
        source='new_rollout',trace=str(path),trace_sha256=sha(path),worker_cpu=list(os.sched_getaffinity(0)),wall_seconds=time.monotonic()-t)
    r=clean(r);cache_writer.append_journal([r],EXP)
    dump(HERE/'raw'/f"{c['rollout_uid']}.json",r);return r

def merge(stage,cases):
    from shared_rollout_db.src import merger
    import sys
    before=list(sys.argv);sys.argv=['merger','--experiment',EXP]
    try:merger.main()
    finally:sys.argv=before
    with db.connect(True)as con:
        for c in cases:
            raw=HERE/'raw'/f"{c['rollout_uid']}.json"
            if not raw.exists():continue
            r=read(raw);row=con.execute('SELECT success,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            assert row and row['success']==r['success']and row['numerical_failure']==r['numerical_failure']and not row['conflict_quarantined'],('DB_READBACK_FAILURE',c['rollout_uid'])

def worker_main(stage,index):
    os.environ.pop('BRD_COORDINATOR',None)
    p=read(HERE/'queues'/f'{stage}.json');cases=p['shards'][index]
    for c in cases:
        if (HERE/'raw'/f"{c['rollout_uid']}.json").exists():continue
        from budget import reserve
        reserve(c)
        r=work(c);print(stage,c['scene'],c['controller_payload']['nominal']['type_code'],c['state_index'],c['solo'],r['outcome'],round(r['wall_seconds'],3),flush=True)

if __name__=='__main__':
    import sys
    worker_main(sys.argv[1],int(sys.argv[2]))
