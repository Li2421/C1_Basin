"""Preregistered bitwise gate design and actual executions, no DB labels."""
from common import *
from actions_c6 import action,case,execute
from schedule_c6 import offline_trigger,TRIGGERS
from progress import offline_switch
from records import actual

def evidence(c):
    with rollout_db.connect(True) as db:r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
    return actual(dict(r),verify=False) if r is not None and not r['conflict_quarantined'] else None

def design():
    prereg();assert read(HERE/'EXECUTION_REPRODUCTION.json')['passed']
    old=D/'crosstype4_no_repair_schedule_diagnosis_v1';train=read(old/'B_training_CASES.json');base=read(HERE/'BASE_PROTOCOL.json');base['states']=read(HERE/'TRAINING_STATES.json')
    if isinstance(base['states'],dict):base['states']=base['states']['states']
    jobs=[];inputs={};sources=[]
    def remember(e):
        for name in ('record_source_path','trace_path'):
            p=Path(e[name]);inputs[str(p)]=dict(path=str(p),sha256=sha(p),size=p.stat().st_size)
    def add(kind,c,s,p,a,ref=None,**kw):
        cc,_=case(p,c['base_controller_name'],s,a,c['seed'],c['state_index'],force_new=True)
        i=len(jobs);cc['rollout_uid']='engineering_'+str(i)+'_'+cc['rollout_uid']
        if ref:remember(ref)
        jobs.append(dict(index=i,kind=kind,case=cc,protocol=p,reference=ref,**kw));return i
    states={s['state_uid']:s for s in base['states']}
    safe=[c for c in train if c['seed']==0 and not np.any(c['eta_a']) and not np.any(c['eta_b'])]
    # Inspect actual trace mechanics only, without selecting by task success.
    safeinfo=[];pertype={}
    for c in safe:
        if pertype.get(c['family'],0)>=16:continue
        e=evidence(c)
        if e is None:continue
        with np.load(e['trace_path'],allow_pickle=False) as z:
            triggers={k:offline_trigger(z['positions'],z['goals'],k) for k in TRIGGERS if k!='none'}
        safeinfo.append((c,e,triggers))
        pertype[c['family']]=pertype.get(c['family'],0)+1
    never=[];prefix=[]
    for c,e,triggers in safeinfo:
        s=states[c['state_uid']]
        if len(never)<8:
            setting=next((k for k,t in triggers.items() if t is None),None)
            if setting:
                a=action([.8,.1,.1],[1.,-.1,.15],.95,setting);add('never_trigger',c,s,base,a,e);never.append(c)
        if len(prefix)<8:
            setting=next((k for k,t in triggers.items() if t is not None),None)
            if setting:
                a=action([.8,.1,.1],[1.,-.1,.15],.95,setting);add('safety_prefix',c,s,base,a,e,expected_trigger=triggers[setting]);prefix.append((c,setting))
        if len(never)==8 and len(prefix)==8:break
    assert len(never)==len(prefix)==8,(len(never),len(prefix))
    # Human threshold identity is distinct: use only mechanically identical switches.
    human=read(old/'B_evaluation_CASES.json');hp=read(old/'B_evaluation_protocol.json');hs={s['state_uid']:s for s in hp['states']};nh=0
    for c in human:
        if c['arm']!='human_progress':continue
        e=evidence(c)
        if e is None:continue
        with np.load(e['trace_path'],allow_pickle=False) as z:
            if offline_switch(z['positions'],z['goals'],700,.968)!=offline_switch(z['positions'],z['goals'],700,.9681620884830543):continue
        add('human_threshold',c,hs[c['state_uid']],hp,action(c['eta_a'],c['eta_b'],.968),e);nh+=1
        if nh==8:break
    assert nh==8
    # Same-trigger constant-early companion physically runs; compare before late lookup.
    for j,(c,setting) in enumerate(prefix):
        a=[.8,.1,.1];s=states[c['state_uid']];trigger=setting if j%2 else 'none'
        companion=add('companion',c,s,base,action(a,a,.95,trigger))
        add('requery_prefix',c,s,base,action(a,[0.,0.,0.],.95,trigger,'requery'),companion=companion)
    # Early constant traces: at least one actual valid reuse in every type.
    singles=[c for c in train if c['seed']==0 and np.array_equal(c['eta_a'],c['eta_b']) and np.any(c['eta_a'])]
    rng=np.random.default_rng(2026102063);rng.shuffle(singles);chosen=set();nr=0
    for c in singles:
        if c['family'] in chosen:continue
        e=evidence(c)
        if e is None:continue
        r=e['record']
        if r['outcome'] not in ('success','collision','numerical_failure') or r['physical_steps']>=600:continue
        with np.load(e['trace_path'],allow_pickle=False) as z:
            if offline_switch(z['positions'],z['goals'],700,.98) is not None:continue
            from progress import progress
            if r.get('error') and progress(z['positions'][-1],z['goals'],np.linalg.norm(z['goals']-z['positions'][0],axis=-1))>=.98:continue
        add('early_reuse',c,states[c['state_uid']],base,action(c['eta_a'],[.7,-.1,.15],.98),e,reuse_rule='constant_early_terminated_before_switch');chosen.add(c['family']);nr+=1
        if len(chosen)==19:break
    for family in sorted(set(base['types'].values())-chosen):
        matches=[x for x in safeinfo if x[0]['family']==family and any(t is None for t in x[2].values())]
        if not matches:continue
        c,e,triggers=matches[0];setting=next(k for k,t in triggers.items() if t is None)
        add('early_reuse',c,states[c['state_uid']],base,action([.8,.1,.1],[.7,-.1,.15],.95,setting),e,reuse_rule='no_trigger_before_termination');chosen.add(family);nr+=1
    assert len(chosen)==19,sorted(set(base['types'].values())-chosen)
    for j in range(13):
        setting=list(TRIGGERS)[1+j%3]
        matches=[x for x in safeinfo if x[2][setting] is None]
        assert matches,setting
        c,e,_=matches[j%len(matches)]
        add('early_reuse',c,states[c['state_uid']],base,action([.8,.1,.1],[.7,-.1,.15],.95,setting),e,reuse_rule='no_trigger_before_termination');nr+=1
    assert nr==32
    write(HERE/'INPUT_ADDITIONS_04.json',dict(utc=utc(),reason='Actual trace sources for mechanical equivalence gates',files=list(inputs.values())))
    write(HERE/'ENGINEERING_CASES.json',jobs)
    manifest=HERE/'engineering_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:j['case'][k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for j in jobs]));write(HERE/'engineering_CACHE_PREFLIGHT.json',planner.preflight(manifest))
    seal('ENGINEERING_DESIGN.receipt.json',[HERE/n for n in ('engineering_gates.py','execution_c6.py','schedule_c6.py','actions_c6.py','ENGINEERING_CASES.json','engineering_CACHE_PREFLIGHT.json','engineering_PLANNED_ROLLOUTS.json')],engineering_only=True,actual_episodes=len(jobs),reuse_validation=32,source_types=sorted(chosen))
    print('Engineering design sealed:',len(jobs),'actual episodes',flush=True)

def run(worker,workers):
    assert not os.environ.get('XLA_FLAGS');rows=read(HERE/'ENGINEERING_CASES.json');cache={}
    for j in rows[worker::workers]:
        path=HERE/f"engineering/raw/{j['index']:03}.json"
        if path.exists():continue
        r=execute(j['case'],j['protocol'],cache,'engineering');write(path,r)
        print(j['index'],j['kind'],r.get('outcome'),r.get('trigger_step'),r.get('progress_switch_step'),flush=True)
    if '_frozen_mean' in cache:cache['_frozen_mean'].close()

def collect():
    rows=read(HERE/'ENGINEERING_CASES.json');checks=[]
    for j in rows:
        if j['kind']=='companion':continue
        r=read(HERE/f"engineering/raw/{j['index']:03}.json");e=j['reference'];kind=j['kind']
        if kind=='requery_prefix':ref=read(HERE/f"engineering/raw/{j['companion']:03}.json");n=r['progress_switch_step']
        else:ref=e['record'];n=r['trigger_step'] if kind=='safety_prefix' else None
        full=kind in ('never_trigger','human_threshold','early_reuse');out={}
        with np.load(ref['trace_path'],allow_pickle=False) as a,np.load(r['trace_path'],allow_pickle=False) as b:
            for k in ('positions','velocities','actions','nominal_actions','goals','projection_active','swept_clearance'):
                x,y=a[k],b[k]
                if not full and k!='goals' and n is not None:
                    stop=n+1 if k in ('positions','velocities') else n;x=x[:stop];y=y[:stop]
                out[k]=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
            if j['case']['trigger']!='none':out['offline_trigger']=offline_trigger(b['positions'],b['goals'],j['case']['trigger'])==r['trigger_step']
        if full:
            for k in ('outcome','episode_length','physical_steps','success','collision','timeout','numerical_failure'):out[k]=ref[k]==r[k]
        if kind=='never_trigger':out['never_trigger']=r['trigger_step'] is None
        if kind=='safety_prefix':out['expected_trigger']=r['trigger_step']==j['expected_trigger']
        if kind=='requery_prefix':out['isolated_requery_prefix']=r['late_eta_requery'] is not None or r['progress_switch_step'] is None
        checks.append(dict(index=j['index'],kind=kind,type=j['case']['family'],trigger=j['case']['trigger'],passed=all(out.values()),checks=out))
    write(HERE/'EXECUTION_EQUIVALENCE.json',dict(utc=utc(),passed=all(x['passed'] for x in checks),checks=checks,failed=[x for x in checks if not x['passed']]))
    early=[x for x in checks if x['kind']=='early_reuse'];write(HERE/'EARLY_TERMINATION_EQUIVALENCE.json',dict(utc=utc(),passed=all(x['passed'] for x in early),validated_types=sorted({x['type'] for x in early if x['passed']}),checks=early))
    seal('EXECUTION_EQUIVALENCE.receipt.json',[HERE/'EXECUTION_EQUIVALENCE.json',HERE/'EARLY_TERMINATION_EQUIVALENCE.json',HERE/'ENGINEERING_DESIGN.receipt.json'])
    print('Equivalence:',len(checks),'checks; failures',sum(not x['passed'] for x in checks),flush=True)

if __name__=='__main__':run(int(sys.argv[2]),int(sys.argv[3])) if sys.argv[1]=='run' else globals()[sys.argv[1]]()
