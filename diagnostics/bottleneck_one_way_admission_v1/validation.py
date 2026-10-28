"""Zero-rollout state-machine checks and full-episode equivalence certificates."""
from dispatch import *
from acceleration import accelerated_case,functions
from types import SimpleNamespace
from admission import AdmissionState,apply
from passages import entry
import shutil

KEYS=('positions','velocities','nominal','safe','actions','projection_rewrite','swept_clearance','clearance_types','route_guides','route_remaining')
OUTCOMES=('success','collision','timeout','numerical_failure','outcome','episode_length','constraint_violation_steps','sampled_speed_violation_steps','restoration_steps','minimum_swept_clearance','final_goal_distances')

def units():
    cfg=SimpleNamespace(agent_radius=.1,max_speed=.5,goal_tolerance=.05)
    passage=dict(centre=[0.,0.],axis=[1.,0.],half_length=.5,width=.3)
    menu=read(HERE/'ONE_WAY_MENU.json')['default'];base=np.array([[.5,0.],[-.5,0.]])
    p=np.array([[-.9,0.],[.9,0.]]);g=-p;rem=np.ones(2)*1.8
    result,_=apply(base,p,g,rem,None,cfg,[],AdmissionState(0),menu)
    assert result.tobytes()==base.tobytes()
    slow=dict(menu,c2=0.,c3=0.);state=AdmissionState(1)
    result,meta=apply(base,p,g,rem,None,cfg,[passage],state,slow)
    assert meta[0]['direction']==1 and np.array_equal(result,np.array([[.5,0.],[0.,0.]]))
    result,meta=apply(base,np.array([[-.55,0.],[.9,0.]]),g,rem,None,cfg,[passage],state,slow)
    assert state.states[0]['count']==1
    # Quota stops further same-direction admissions and grants the next turn
    # to the opposite side only after current-direction occupants leave.
    state.states[0].update(direction=1,count=3,previous_inside=[False,False])
    quota=dict(slow,K=3)
    _,meta=apply(base,p,g,rem,None,cfg,[passage],state,quota)
    assert meta[0]['direction']==-1
    # Opposing robot already inside exits with the admitted direction.
    state=AdmissionState(1);state.states[0]['direction']=1
    target,meta=apply(np.array([[.2,0.],[-.4375,.1]]),np.array([[-.2,0.],[.2,0.]]),np.array([[1.,0.],[-1.,0.]]),rem,None,cfg,[passage],state,menu)
    assert meta[0]['reverse'][1] and np.array_equal(target[1],np.array([.5,0.]))
    assert not any(meta[0]['clear_exit'])
    # A goal-complete robot at the exit still receives clearance motion.
    state=AdmissionState(1);state.states[0]['direction']=1
    target,meta=apply(np.zeros((2,2)),np.array([[-.2,0.],[.7,0.]]),np.array([[1.,0.],[.7,0.]]),np.array([1.,0.]),None,cfg,[passage],state,menu)
    assert meta[0]['clear_exit'][1] and np.linalg.norm(target[1])>0
    state=AdmissionState(1);state.states[0]['direction']=1
    target,meta=apply(np.zeros((2,2)),np.array([[-.2,0.],[.55,0.]]),np.array([[1.,0.],[.55,0.]]),np.array([1.,0.]),None,cfg,[passage],state,menu)
    assert meta[0]['geometry_inside'][1] and not meta[0]['clear_exit'][1] and np.array_equal(target[1],np.zeros(2))
    assert entry(np.array([[-2.,0.],[-1.,0.]]),passage,.1,.8)is None
    from runtime import ROUTE_EVALUATE,RoleState
    cfg.agent_collision_margin=0.
    filters=[SimpleNamespace(A=np.zeros((0,2)),lower=[],kinds=[]),
             SimpleNamespace(A=np.array([[1.,0.]]),lower=[1.],kinds=[['pair',1,2]]),
             SimpleNamespace(A=np.array([[-1.,0.]]),lower=[1.],kinds=[['pair',2,1]])]
    p=np.array([[0.,0.],[1.,0.],[1.21,0.]])
    for allowed in [False,True]:
        role=RoleState(pair=(0,1),passer=0,yielder=1,activation_step=0)
        ROUTE_EVALUATE(p,p+1.,np.zeros_like(p),np.zeros_like(p),np.zeros((3,3,2)),filters,[],cfg,role,names=['Y1'],route_remaining=np.ones(3),allow_reselection=allowed)
        assert role.pair==((1,2)if allowed else(0,1))
    dump(HERE/'ADMISSION_UNIT_CHECK.json',dict(passed=True,definition_version=2,environment_steps=0,checks=['no-passage bit identity','FCFS tie robot ID','waiting target cancellation','entry count','quota opposite turn','inverse inside c1=c3=1 produces exact exit vector','no geometric-inside robot receives clearance including arrived agents','arrived outside exit blocker clearance','finite route lookahead','inactive role resets only in explicit ablation']))

def acceleration_units(records):
    path=HERE/'ACCELERATION_UNITS.json'
    if path.exists()and read(path)['passed']:return
    import importlib.util
    spec=importlib.util.spec_from_file_location('local_threefry',HERE/'accel_inputs/np_threefry.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rng=np.random.default_rng(2026102903)
    for _ in range(3000):
        key=rng.integers(0,2**32,2,dtype=np.uint32);data=int(rng.integers(0,2**32,dtype=np.uint64))
        assert np.array_equal(mod.fold_in(key,data),np.asarray(RN.jax.random.fold_in(key,data)))
    chains=0
    for item in records:
        c=item['case'];root=RN.jax.random.fold_in(RN.jax.random.fold_in(RN.jax.random.PRNGKey(2026102004),int(hashlib.sha256(c['rng_state_uid'].encode()).hexdigest()[:8],16)),c['seed'])
        # Entire actual horizon for the shared joint random stream used by
        # these two frozen nominal controllers, not only an early prefix.
        for t in range(c['scenario_payload']['config']['max_steps']):
            a=mod.fold_in(root,t);b=RN.jax.random.fold_in(root,t)
            assert np.array_equal(a,np.asarray(b));chains+=1
    import shared_control.hard_projection as HP
    import clarabel
    from scipy import sparse
    ns=dict(np=np,HP=HP,clarabel=clarabel,sparse=sparse,_P={})
    functions(HERE/'accel_inputs/fastproj.py',{'fast_project_velocity'},ns)
    errors=0
    for j in range(10000):
        n=1+j%3;matrix=rng.normal(size=(j%9+1,2*n));matrix[rng.random(matrix.shape)<.2]=0.
        lower=-rng.uniform(.1,1.,len(matrix));target=rng.normal(size=(n,2));target[0]=[1.,0.]
        try:ns['fast_project_velocity'](target,matrix,lower,.5,HP.HardProjectionConfig(),check=True)
        except HP.CBFSolverError:errors+=1
    dump(path,dict(passed=True,random_inputs=3000,actual_chain_comparisons=chains,csc_checks=10000,synthetic_unknowns=errors,environment_steps=0))

def compare(a,b):
    with np.load(a['trace'])as x,np.load(b['trace'])as y:
        arrays={k:x[k].shape==y[k].shape and x[k].dtype==y[k].dtype and x[k].tobytes()==y[k].tobytes()for k in KEYS}
    outcomes={k:a.get(k)==b.get(k)for k in OUTCOMES}
    return dict(equal=all(arrays.values())and all(outcomes.values()),arrays=arrays,outcomes=outcomes,a=a['rollout_uid'],b=b['rollout_uid'])

def select_replays():
    path=HERE/'REPLAY_INPUTS.json'
    if path.exists():
        records=read(path)['records']
        assert all(x['case']['controller_payload']['execution_sources']==sources(x['case']['spec'])for x in records),'Replay requests must match frozen current numerical sources'
        return records
    state=read(HERE/'inputs/opening/SEARCH_STATE.json');scenes={e['scene']['name']:e['scene']for e in read(HERE/'SCENES.json')['entries']}
    ctls={c['type_code']:c for c in controllers()};oldroot=ROOT/'diagnostics/scene_types_oracle_coverage_v3'
    buckets={};seen=set()
    # Fully saved exact six-times-horizon records only; avoid transferring an
    # older timeout trajectory into a longer-horizon replay certificate.
    for g in state['groups']:
        for q in g['queries']:
            uid=q['target_query_uid'];raw=oldroot/'raw'/f'{uid}.json'
            if uid in seen or not raw.exists():continue
            seen.add(uid);key=(g['type_id'],g['controller'],q['outcome'])
            if len(buckets.setdefault(key,[]))<12:buckets[key].append((g,raw))
    selected=[]
    for index in range(12):
        for key in sorted(buckets):
            if len(buckets[key])>index:selected.append(buckets[key][index])
            if len(selected)>=200:break
        if len(selected)>=200:break
    assert len(selected)==200 and len({g['type_id']for g,p in selected})==17
    rows=[]
    for g,raw in selected:
        r=read(raw);dest=HERE/'inputs/replays'/raw.name;dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(raw.read_bytes());trace=Path(r['trace']);copytrace=dest.with_suffix('.npz');shutil.copyfile(trace,copytrace)
        r=dict(r,trace=str(copytrace))
        c=accelerated_case(scenes[g['scene']],ctls[g['controller']],g['state_index'],r['spec'],purpose='old_equivalence')
        rows.append(dict(case=c,reference=r,source_raw=str(raw),source_sha256=sha(raw),source_trace=str(trace),trace_sha256=sha(copytrace),type_id=g['type_id']))
    dump(path,dict(records=rows,count=len(rows),selection='Round robin over type/controller/outcome; all17 types, both nominals, exact saved full trajectories only.'))
    return rows

def certify():
    units()
    if (HERE/'EQUIVALENCE.json').exists()and not read(HERE/'EQUIVALENCE.json')['passed']:
        raise RuntimeError('HARD_STOP old trajectory equivalence previously failed; inspect before retry')
    records=select_replays()
    acceleration_units(records)
    if not(HERE/'EQUIVALENCE.json').exists():
        rows={r['rollout_uid']:r for r in execute([x['case']for x in records],'accel_paired_old200',worker_entry='worker_accel.py')}
        checks=[dict(type_id=x['type_id'],controller=x['case']['controller_payload']['nominal']['type_code'],**compare(x['reference'],rows[x['case']['rollout_uid']]))for x in records if x['case']['rollout_uid']in rows]
        passed=len(checks)==200 and all(x['equal']for x in checks)
        dump(HERE/'EQUIVALENCE.json',dict(passed=passed,episodes=len(checks),checks=checks,route_modes=sorted({x['case']['scenario_payload']['route_guidance']for x in records})))
        if not passed:raise RuntimeError('HARD_STOP old candidate bitwise equivalence failed')
    if (HERE/'ACCELERATION_CERTIFICATE.json').exists():return read(HERE/'ACCELERATION_CERTIFICATE.json')['passed']
    scenes=[e['scene']for e in read(HERE/'SCENES.json')['entries']if read(HERE/'PASSAGES.json')['by_scene'][e['scene']['name']]['representative']and read(HERE/'PASSAGES.json')['by_scene'][e['scene']['name']]['passages']]
    fixed=read(HERE/'FAMILIES_THIS_ROUND.json')['fixed_admission'];pairs=[]
    for si in range(3):
        for sc in scenes:
            for ctl in controllers():
                pairs.append((case(sc,ctl,si,fixed,purpose='new_acceleration_pair'),accelerated_case(sc,ctl,si,fixed,purpose='new_acceleration_pair')))
    pairs=pairs[:50];assert len(pairs)==50
    originals={r['rollout_uid']:r for r in execute([a for a,b in pairs],'new_columns_original50',worker_entry='worker_recorded.py')}
    fast={r['rollout_uid']:r for r in execute([b for a,b in pairs],'accel_paired_new50',worker_entry='worker_accel.py')}
    checks=[compare(originals[a['rollout_uid']],fast[b['rollout_uid']])for a,b in pairs if a['rollout_uid']in originals and b['rollout_uid']in fast]
    passed=len(checks)==50 and all(c['equal']for c in checks)
    oldinputs=read(HERE/'inputs/ACCELERATION_INPUTS.json')['files']
    hashes_equal=all(sha(HERE/'accel_inputs'/Path(x['target']).name)==x['sha256']for x in oldinputs)
    result=dict(passed=passed and hashes_equal,old_replays=200,new_pairs=len(checks),checks=checks,acceleration_inputs_hash_equal=hashes_equal,
                original_seconds=sum(r['wall_seconds']for r in originals.values()),fast_seconds=sum(r['wall_seconds']for r in fast.values()),
                enabled=['same_CSC_construction','numpy_threefry_fold','numpy_far_wall_cost_pruning','local_far_wall_prescreen']if passed and hashes_equal else [],
                units_reference='inputs/ACCELERATION_CERTIFICATE.json: prior 3000 random keys/10000 CSC checks; local additional checks recorded separately')
    dump(HERE/'ACCELERATION_CERTIFICATE.json',result)
    if not result['passed']:
        with(HERE/'ITERATION_LOG.md').open('a')as f:f.write('\n新列加速对跑不一致，正式任务关闭加速，使用原算法。\n')
    return result['passed']
