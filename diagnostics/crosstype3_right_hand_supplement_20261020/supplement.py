from common import *
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity,transaction
import copy,subprocess,time,statistics
_original_write=write
def write(path,value):
    path=Path(path)
    return _original_write(path if path.is_absolute() else HERE/path,value)
SRC=D/'crosstype3_unseen_controller_schedule_v1';C2=D/'crosstype2_mixed_zoo_selector_v1'
TYPES=('transformer_bc','gradient_mpc','conflict_based_search')
RULE=dict(mode='progress',steps=40,speed=.35)
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def design():
    prereg();p=read(SRC/'new_evaluation_protocol.json');old=read(SRC/'new_evaluation_CASES.json')
    keys={}
    for c in old:
        if c['family'] in TYPES:keys.setdefault((c['base_controller_name'],c['state_uid'],c['seed']),c)
    assert len(keys)==1728
    rows=[];profiles={}
    for key,c0 in sorted(keys.items()):
        c=copy.deepcopy(c0);name=c['base_controller_name'];v=copy.deepcopy(p['profiles'][name]['payload'])
        for k in ('generator_schedule','eta_schedule','progress_schedule','generator_progress_schedule'):
            assert k not in v,k
        v['continuing_right_hand']=dict(rule=RULE,implementation_sha256=sha(HERE/'right_hand.py'),trigger_sha256=sha(HERE/'trigger.py'))
        cid=uid('ctl',v);eid=eta_identity([0.,0.,0.])[0];sk=c['seed_key']
        c.update(controller_uid=cid,eta_uid=eid,rollout_uid=uid('roll',dict(state=c['state_uid'],eta=eid,controller=cid,seed=sk)),arm='RHR',candidate_index=13,eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],switch_mode='fixed',switch_step=700,rhr_rule=RULE)
        rows.append(c);profiles[cid]=dict(controller_uid=cid,payload=v)
    p['phase']='evaluation';p['right_hand_supplement']=dict(rule=RULE,parent=str(SRC))
    inserted=[]
    with rollout_db.connect() as db,transaction(db):
        def ins(table,values):
            key=next(iter(values));oldrow=db.execute('SELECT * FROM '+table+' WHERE '+key+'=?',(values[key],)).fetchone()
            if oldrow is not None:
                assert all(oldrow[k]==v for k,v in values.items()),(table,values[key]);return
            db.execute('INSERT OR IGNORE INTO '+table+'('+','.join(values)+') VALUES('+','.join('?' for _ in values)+')',list(values.values()));inserted.append((table,values[key]))
        ins('experiment',dict(experiment_uid=EXP,name='Three unseen controller fixed right hand supplement',path=str(HERE),protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=(db.execute('SELECT code_hash FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone() or [sha(__file__)])[0],metadata_json=canonical(dict(user_authorized_cpu_loan=True,parent=str(SRC)))))
        for c in rows:
            assert db.execute('SELECT 1 FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone()
        for cid,r in profiles.items():
            v=r['payload'];ins('controller_config',dict(controller_uid=cid,scenario_uid=v['scenario_uid'],flow_checkpoint_sha256=v.get('flow_checkpoint_sha256'),orthoflow3_sha256=v['orthoflow3_sha256'],safety_config_hash=uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),horizon=str(v['horizon']),dt=str(v['dt']),success_semantics_version=v['success_semantics'],conditioning_version=v['conditioning'],rng_semantics_version=canonical(v['rng']),config_json=canonical(v),compatibility_quality='EXACT_PROFILE'))
        assert db.execute('SELECT 1 FROM eta WHERE eta_uid=?',(rows[0]['eta_uid'],)).fetchone()
    write('REGISTRATION.json',dict(utc=utc(),inserted=inserted,preexisting_rows_unchanged=True))
    write('evaluation_protocol.json',p);write('evaluation_CASES.json',rows)
    write('evaluation_PLANNED_ROLLOUTS.json',dict(requests=[{k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for c in rows]))
    pre=planner.preflight(HERE/'evaluation_PLANNED_ROLLOUTS.json');assert pre['summary']['ambiguous']==0
    write('evaluation_CACHE_PREFLIGHT.json',pre)
    missing={(r['state_uid'],r['eta_uid'],r['controller_uid'],s) for r in pre['details'] for s in r['missing_seeds']}
    with rollout_db.connect(True) as db:
        todo=[c for c in rows if (c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key']) in missing and db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None]
    write('evaluation_MISSING_CASES.json',todo)
    inputs={str(SRC/'new_evaluation_protocol.json'):sha(SRC/'new_evaluation_protocol.json'),str(SRC/'new_evaluation_CASES.json'):sha(SRC/'new_evaluation_CASES.json')}
    for r in profiles.values():
        path=r['payload'].get('flow_checkpoint_path')
        if path:inputs[path]=sha(Path(path))
    write('INPUT_HASHES.json',inputs)
    seal('evaluation_EXECUTION_FROZEN.receipt.json',[*HERE.glob('*.py'),*HERE.glob('*.so'),HERE/'evaluation_protocol.json',HERE/'evaluation_CASES.json',HERE/'evaluation_MISSING_CASES.json',HERE/'PREREGISTRATION.md'],before_any_new_evaluation_rollout=True,preflight_summary=pre['summary'],requested=len(rows),genuinely_missing=len(todo))
    print(json.dumps(pre['summary']),flush=True)

def reproduce():
    from new_execution import execute
    tests=[]
    # One actual historical RHR episode plus Safety for each new type; no tuning.
    candidates=read(C2/'new_evaluation_CASES.json');r=next(c for c in candidates if c['arm']=='RHR' and c['family']=='Flow')
    tests.append((r,read(C2/'new_evaluation_protocol.json'),C2/'new_evaluation'/'raw'/f"{r['rollout_uid']}.json"))
    old=read(SRC/'new_evaluation_CASES.json')
    for kind in TYPES:
        for c in old:
            if c['family']==kind and c['arm']=='safety_baseline':
                f=SRC/'new_evaluation'/'raw'/f"{c['rollout_uid']}.json"
                if f.exists():tests.append((c,read(SRC/'new_evaluation_protocol.json'),f));break
        else:raise RuntimeError('No old safety case '+kind)
    checks=[]
    for i,(c,p,path) in enumerate(tests):
        source=read(path);assert not source['numerical_failure']
        out=execute(c,p,{},'reproduction');oldtrace=np.load(source['trace_path']);newtrace=np.load(out['trace_path'])
        equal={k:bool(np.array_equal(oldtrace[k],newtrace[k])) for k in ('positions','actions','velocities')}
        assert all(equal.values()) and out['outcome']==source['outcome'] and out['episode_length']==source['episode_length'],(i,equal,out['outcome'],source['outcome'])
        checks.append(dict(source=str(path),rollout_uid=c['rollout_uid'],outcome_equal=True,episode_length=out['episode_length'],bitwise=equal))
        write(f'reproduction/result_{i}.json',out);print('reproduction',i,'passed',flush=True)
    write('REPRODUCTION.json',dict(utc=utc(),passed=True,checks=checks,engineering_only=True))

def worker(index):
    import fcntl
    from new_execution import execute
    for path,h in read(HERE/'evaluation_EXECUTION_FROZEN.receipt.json')['files'].items():assert sha(path)==h,path
    assert read(HERE/'REPRODUCTION.json')['passed']
    p=read(HERE/'evaluation_protocol.json');rows=read(HERE/'evaluation_MISSING_CASES.json');cache={};folder=HERE/'evaluation';folder.mkdir(exist_ok=True)
    while True:
        with (folder/'cursor').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);i=int(f.read().strip() or 0);f.seek(0);f.truncate();f.write(str(i+1));f.flush()
        if i>=len(rows):break
        c=rows[i];raw=folder/'raw'/f"{c['rollout_uid']}.json"
        if raw.exists():continue
        with rollout_db.connect(True) as db:
            if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone():continue
        out=execute(c,p,cache,'evaluation');write(raw,out);cache_writer.append_journal([out],EXP,f'evaluation_{index}')
        if out['error'] is not None and out['error']['type'] not in ('CBFSolverError','FloatingPointError'):raise RuntimeError(str(out['error']))
        print(json.dumps(dict(utc=utc(),worker=index,index=i,total=len(rows),seconds=out['elapsed_seconds'],error=out['error'])),flush=True)
    write(f'evaluation/WORKER_{index}_COMPLETE.json',dict(utc=utc(),worker=index))

if __name__=='__main__':
    if sys.argv[1]=='design':design()
    elif sys.argv[1]=='reproduce':reproduce()
    elif sys.argv[1]=='worker':worker(int(sys.argv[2]))
