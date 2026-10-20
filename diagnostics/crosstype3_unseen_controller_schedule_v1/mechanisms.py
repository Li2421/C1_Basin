"""The original24 outcome-free mechanism features, with trace-level reuse checks."""
from common import *
from features import NAMES,aggregate
ARMS=('safety_baseline','F13','F10')

def historical_entries():
    return [r for r in read(D/'crosstype2_mixed_zoo_selector_v1/PROBE_TRAJECTORY_INDEX.json') if r['arm'] in ARMS]

def compute(name,entries,randomfile,cfg):
    traces={a:[None]*16 for a in ARMS};bad=[]
    for e in entries:
        if e['controller']!=name:continue
        raw=Path(e['raw_path']);assert sha(raw)==e['raw_sha256'];r=read(raw)
        assert not r['terminal_labels_read'] and not any(k in r for k in ('success','outcome','termination'))
        assert sha(e['trace_path'])==e['trace_sha256']
        if not r['complete']:bad.append(dict(arm=e['arm'],state=e['state_index']))
        with np.load(e['trace_path'],allow_pickle=False) as z:traces[e['arm']][e['state_index']]={k:z[k] for k in z.files}
    if bad or any(t is None for arm in ARMS for t in traces[arm]):return np.full(24,np.nan),bad
    receipt=read(randomfile.with_suffix('.receipt.json'));assert sha(randomfile)==receipt['sha256']
    with np.load(randomfile,allow_pickle=False) as z:actions=z['actions']
    return aggregate(traces,actions,cfg['dt'],cfg['max_speed'],cfg['goal_tolerance']),bad

def old_random(name):
    for root,phases in [(D/'crosstype1_learned_zoo_selector_v1',('old_probe','new_probe','locked_probe')),(D/'crosstype2_mixed_zoo_selector_v1',('probe',))]:
        for phase in phases:
            path=root/phase/'initial_actions'/f'{name}.npz'
            if path.exists():return path
    raise FileNotFoundError(name)

def reuse():
    base=read(HERE/'BASE_PROTOCOL.json');names=base['controller_order'];cfg=base['native_manifest']['scenario_config'];entries=historical_entries();opening=read(HERE/'OPENING_CONFIG.json')
    with np.load(D/'crosstype1c_two_eta_schedule_v1/CONTROLLER_FEATURES.npz',allow_pickle=False) as z:old1={str(n):x for n,x in zip(z['names'],z['features'])}
    with np.load(D/'crosstype2_mixed_zoo_selector_v1/CONTROLLER_FEATURES.npz',allow_pickle=False) as z:old2={str(n):x[:24] for n,x in zip(z['names'],z['features'])}
    checks=[]
    for cohort,selected,reference in [('crosstype1c',opening['mechanism_source_recompute_c1c'],old1),('crosstype2',opening['mechanism_source_recompute_c2'],old2)]:
        for name in selected:
            v,bad=compute(name,entries,old_random(name),cfg);target=reference[name]
            checks.append(dict(cohort=cohort,controller=name,bitwise=v.tobytes()==target.tobytes(),max_abs_error=float(np.nanmax(abs(v-target))),incomplete=bad))
    passed=all(c['bitwise'] for c in checks);values=[];gaps=[]
    for name in names:
        if passed:v=old1[name] if name in old1 else old2[name]
        else:v,bad=compute(name,entries,old_random(name),cfg)
        values.append(v)
        if not np.isfinite(v).all():gaps.append(dict(controller=name,reason='Historical incomplete probe; NaN kept; no unknown rerun'))
    np.savez_compressed(HERE/'TRAINING_CONTROLLER_FEATURES.npz',names=names,types=[base['types'][n] for n in names],features=values,feature_names=NAMES)
    write(HERE/'FEATURE_REUSE_CHECK.json',dict(utc=utc(),checks=checks,all_eight_bitwise=passed,decision='Reuse exact24 values' if passed else 'Recomputed all104 from historical compatible traces',gaps=gaps,rhr_traces_excluded=True))
    write(HERE/'HISTORICAL_MECHANISM_TRACE_INDEX.json',entries)
    seal('TRAINING_CONTROLLER_FEATURES.receipt.json',[HERE/'TRAINING_CONTROLLER_FEATURES.npz',HERE/'FEATURE_REUSE_CHECK.json',HERE/'HISTORICAL_MECHANISM_TRACE_INDEX.json',HERE/'mechanisms.py',HERE/'features.py'],outcome_labels_read=False)
    print('Historical mechanism checks:',sum(c['bitwise'] for c in checks),'/8 bitwise;',len(gaps),'preserved missing vectors',flush=True)

def initial():
    from new_execution import ensure
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    import jax
    jax.config.update('jax_enable_x64',False)
    p=read(HERE/'probe_protocol.json');cache={}
    for name in p['controller_order']:
        path=HERE/'probe/initial_actions'/f'{name}.npz';receipt=path.with_suffix('.receipt.json')
        if receipt.exists():assert sha(path)==read(receipt)['sha256'];continue
        ensure(dict(base_controller_name=name),p,cache);rt=cache[name];actions=[]
        for s in p['states']:
            current=[]
            for seed in range(8):
                env=rt.make_env(s);_,action,_=rt.nominal(env,key_for(s,seed,0));assert env.step_count==0;current.append(np.asarray(action).copy())
            actions.append(current)
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,actions=np.asarray(actions))
        write(receipt,dict(utc=utc(),sha256=sha(path),controller=name,physical_steps=0,terminal_labels_read=False,checkpoint_sha256=p['profiles'][name]['payload'].get('flow_checkpoint_sha256'),source_code_sha256=sha(__file__)))
        print('Initial no-physics nominal queries:',name,flush=True)

def assemble():
    assert (HERE/'probe_COMPLETE.receipt.json').exists();p=read(HERE/'probe_protocol.json');cfg=p['native_manifest']['scenario_config'];entries=[]
    for c in read(HERE/'probe_CASES.json'):
        path=HERE/'probe/raw'/f"{c['rollout_uid']}.json";r=read(path)
        entries.append(dict(controller=c['base_controller_name'],arm=c['arm'],state_index=c['state_index'],raw_path=str(path),raw_sha256=sha(path),trace_path=r['trace_path'],trace_sha256=r['trace_sha256'],complete=r['complete']))
    with np.load(HERE/'TRAINING_CONTROLLER_FEATURES.npz',allow_pickle=False) as z:names=z['names'].tolist();types=z['types'].tolist();values=list(z['features'])
    gaps=[]
    for name in p['controller_order']:
        value,bad=compute(name,entries,HERE/'probe/initial_actions'/f'{name}.npz',cfg);names.append(name);types.append(p['types'][name]);values.append(value)
        if bad or not np.isfinite(value).all():gaps.append(dict(controller=name,incomplete_probe_arms=bad,reason='No unknown rerun or partial averaging; source-only fill during critic inference'))
    np.savez_compressed(HERE/'CONTROLLER_FEATURES.npz',names=names,types=types,features=values,feature_names=NAMES)
    write(HERE/'UNSEEN_MECHANISM_TRACE_INDEX.json',entries);write(HERE/'UNSEEN_FEATURE_GAPS.json',gaps)
    seal('CONTROLLER_FEATURES.receipt.json',[HERE/'CONTROLLER_FEATURES.npz',HERE/'UNSEEN_MECHANISM_TRACE_INDEX.json',HERE/'UNSEEN_FEATURE_GAPS.json',HERE/'mechanisms.py',HERE/'features.py',HERE/'TRAINING_CONTROLLER_FEATURES.receipt.json'],outcome_labels_read=False)

if __name__=='__main__':globals()[sys.argv[1]]()
