"""Production-source adapter and cache-safe execution, CPU only."""
import os
AVAILABLE_CPUS=sorted(os.sched_getaffinity(0))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu'
import sys,json,hashlib,time,multiprocessing as mp,subprocess
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;OLD=ROOT/'diagnostics/b5_oracle_20261027'
sys.path.insert(0,str(OLD));sys.path.insert(0,str(ROOT))
import runtime as old_runtime
import oracle as old_oracle
os.sched_setaffinity(0,AVAILABLE_CPUS)  # Undo the historical orchestrator's old 18-CPU cap.
from shared_control.basis_dictionary import evaluate,add_target,RoleState,NEW,ALL
from shared_rollout_db.src import rollout_db as db,planner,cache_writer
import numpy as np
RN=old_runtime.RN;inputs=old_oracle.inputs;clean=old_runtime.clean;dump=old_runtime.dump
EXP='exp_basis_discovery_20261027'
ANCHOR='            second=[f.project(target[i:i+1])'
assert old_runtime.SOURCE.count(ANCHOR)==1
ADAPTED=old_runtime.SOURCE.replace('def run(c):','def run(c, hook):').replace(ANCHOR,'            target=hook(env,v,ref,filters,t,eta,target)\n'+ANCHOR)
ns=dict(vars(RN));exec(compile(ADAPTED,str(OUT/'generated_hook.py'),'exec'),ns);RUN=ns['run']


def digest(x):return hashlib.sha256(db.canonical(clean(x)).encode()).hexdigest()
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@lru_cache(None)
def historical():
    rows=list(map(json.loads,(OLD/'candidates_verified.jsonl').open()))
    return [r for r in rows if 'scene'in r]


def behavior(spec):
    coefficients={k:float(v)for k,v in spec.get('coefficients',{}).items()if v!=0.}
    b=dict(eta=list(map(float,spec['eta'][:3])),coefficients=coefficients)
    if 'T5'in coefficients:b['t5_seed']=spec.get('t5_seed','tangent')
    if 'Y3'in coefficients:b['clearance_side']=int(spec.get('clearance_side',1))
    if spec.get('schedule'):b['schedule']=spec['schedule']
    if spec.get('rhr'):b['rhr']=spec['rhr']
    return b


def make_case(scene,controller,si,spec,seed=0):
    spec=behavior(spec)
    c=old_runtime.make_case(scene,controller,si,dict(spec,mode='dictionary'),seed)
    c['spec']=spec
    c['controller_payload'].update(dictionary=spec,dictionary_sources={
        'basis_dictionary':file_sha(ROOT/'shared_control/basis_dictionary.py'),'runner':file_sha(__file__)},
        correction_semantics='dictionary_v1_original_GPR_original_safety_original_nominal')
    c['controller_uid']=db.uid('ctl',dict(scenario=c['scenario_uid'],payload=c['controller_payload']))
    c['rollout_uid']=db.uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))
    return c


class Hook:
    def __init__(self,spec):
        self.spec=spec;self.role=RoleState();self.calls=0;self.seconds=0.;self.svd_seconds=0.;self.svd_rank_counts=[0,0,0]
        self.unavailable={k:0 for k in spec.get('coefficients',{})};self.max_norm={k:0. for k in self.unavailable};self.degenerate={}
    def __call__(self,env,safe,nominal,filters,t,eta,original):
        coefficients=self.spec.get('coefficients',{})
        if not coefficients:return original
        if self.spec.get('schedule')=='release_third'and t>=env.config.max_steps//3:return original
        cfg=env.config
        fields=np.stack([RN.distributed_fields(env.positions,env.goals,safe,i,cfg.max_speed,cfg.goal_tolerance,RN.contact_radius(cfg))for i in range(env.num_agents)])
        columns,meta=evaluate(env.positions,env.goals,safe,nominal,fields,filters,env.snapshot()['walls'],cfg,self.role,t,
            names=coefficients,t5_seed=self.spec.get('t5_seed','tangent'),clearance_side=self.spec.get('clearance_side',1))
        self.calls+=1;self.seconds+=meta['seconds'];self.svd_seconds+=meta['svd_seconds']
        for rank in meta['svd_ranks']:self.svd_rank_counts[rank]+=1
        for k,v in meta['unavailable'].items():self.unavailable[k]+=int(v);self.max_norm[k]=max(self.max_norm[k],float(np.linalg.norm(columns[k])))
        for k,v in meta['degenerate'].items():self.degenerate[k]=self.degenerate.get(k,0)+int(v)
        return add_target(safe,fields,columns,eta,coefficients)
    def stats(self):return dict(calls=self.calls,seconds=self.seconds,svd_seconds=self.svd_seconds,svd_rank_counts=self.svd_rank_counts,
        unavailable_steps=self.unavailable,max_column_norm=self.max_norm,degenerate_steps=self.degenerate,role=self.role.__dict__)


def run(c):
    hook=Hook(c['spec'])
    if c['spec'].get('rhr'):
        from rhr_exact import run as rhr_run
        cc=dict(c,spec=dict(c['spec'],mode='exact_rhr_centre'if c['spec']['rhr']=='B'else'exact_rhr_goal'))
        result,arrays=rhr_run(cc)
    else:result,arrays=RUN(c,hook)
    cfg=c['scenario_payload']['config']
    result.update(completion_time=result['episode_length']*cfg['dt']if result['success']else None,
        truncated_time=result['episode_length']*cfg['dt']if result['success']else cfg['dt']*cfg['max_steps'],
        horizon_time=cfg['dt']*cfg['max_steps'],dictionary=hook.stats())
    return clean(result),arrays


def register(cases,stage):
    with db.connect()as con,db.transaction(con):
        con.execute('INSERT INTO experiment(experiment_uid,name,path,metadata_json)VALUES(?,?,?,?) ON CONFLICT(experiment_uid) DO NOTHING',
            (EXP,EXP,str(OUT),db.canonical(dict(phase='dictionary_screen_no_training'))))
        for c in cases:
            physical={k:c['state'][k]for k in ('positions','velocities','goals')}
            con.execute('INSERT INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json)VALUES(?,?,?,?) ON CONFLICT(scenario_uid) DO NOTHING',
                (c['scenario_uid'],c['scene'],c['scenario_uid'],db.canonical(c['scenario_payload'])))
            con.execute('INSERT INTO state(state_uid,scenario_uid,content_hash,physical_state_json,provenance_json,identity_quality)VALUES(?,?,?,?,?,?) ON CONFLICT(state_uid) DO NOTHING',
                (c['state_uid'],c['scenario_uid'],digest(physical),db.canonical(physical),db.canonical(dict(experiment=EXP,state_index=c['state_index'])),'CONTENT_EXACT'))
            assert con.execute('SELECT physical_state_json FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone()[0]==db.canonical(physical)
            con.execute('INSERT INTO controller_config(controller_uid,scenario_uid,config_json,compatibility_quality)VALUES(?,?,?,?) ON CONFLICT(controller_uid) DO NOTHING',
                (c['controller_uid'],c['scenario_uid'],db.canonical(c['controller_payload']),'EXACT_PROFILE'))
            assert con.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(c['controller_uid'],)).fetchone()[0]==db.canonical(c['controller_payload'])
            eid,e,hx=db.eta_identity(c['eta_a']);con.execute('INSERT INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex)VALUES(?,?,?,?,?) ON CONFLICT(eta_uid) DO NOTHING',(eid,*e,hx))
    path=OUT/f'{stage}_planned.json';dump(path,dict(requests=[{k:c[k]for k in ('state_uid','eta_uid','controller_uid','seed_key')}for c in cases],exact_configs=cases))
    pf=planner.preflight(path);dump(OUT/f'{stage}_preflight.json',pf);assert pf['summary']['ambiguous']==0
    return {(x['state_uid'],x['eta_uid'],x['controller_uid'],s)for x in pf['details']for s in x['missing_seeds']}


def init_worker():
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[cpus[(mp.current_process()._identity[0]-1)%len(cpus)]])


def work(c):
    t0=time.monotonic();result,arrays=run(c);path=OUT/'traces'/f"{c['rollout_uid']}.npz";path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,**arrays)
    r=dict(result,**{k:c[k]for k in ('scenario_uid','state_uid','controller_uid','eta_uid','seed_key','rollout_uid','scene','state_index','seed','spec')},
        controller=c['controller_payload']['nominal']['type_code'],eta=c['eta_a'],future_index=c['seed'],
        cache_identity_schema='registered_exact_uid_v1',initial_positions=c['state']['positions'],initial_velocities=c['state']['velocities'],goals=c['state']['goals'],
        deadlock=0,source='new_rollout',trace=str(path),worker_cpu=list(os.sched_getaffinity(0)),wall_seconds=time.monotonic()-t0)
    cache_writer.append_journal([r],EXP);return r


def exact_old_key(r):
    sp=r['spec'];e=[v if i in sp.get('keep',range(5))else 0. for i,v in enumerate(sp['eta'])]
    if sp.get('schedule')or sp.get('theta')is not None:return None
    if sp['mode']not in ('b3','add','blend')or e[4]!=0:return None
    co={}
    if e[3]!=0:
        if sp.get('recipe')not in ('goal','cluster'):return None
        co={'T1'if sp['recipe']=='goal'else'T2':e[3]}
    return (r['scene'],r['state_index'],r['controller'],r['seed'],digest(behavior(dict(eta=e[:3],coefficients=co))))

@lru_cache(None)
def old_cache():return {key:r for r in historical()if not r.get('numerical_failure')and (key:=exact_old_key(r))is not None}


def execute(cases,stage,workers=None):
    cases=list({c['rollout_uid']:c for c in cases}.values());missing=register(cases,stage)
    path=OUT/'candidates.jsonl';done={r['rollout_uid']:r for r in map(json.loads,path.open())}if path.exists()else{}
    todo=[];reused=0;t0=time.monotonic()
    with path.open('a')as f:
        for c in cases:
            if c['rollout_uid']in done:continue
            key=(c['scene'],c['state_index'],c['controller_payload']['nominal']['type_code'],c['seed'],digest(c['spec']))
            old=old_cache().get(key)
            if old:
                r=dict(old,**{k:c[k]for k in ('scenario_uid','state_uid','controller_uid','eta_uid','seed_key','rollout_uid','scene','state_index','seed','spec')},
                    eta=c['eta_a'],future_index=c['seed'],source='exact_prior_behavior_reuse',source_rollout_uid=old['rollout_uid'],wall_seconds=0.,dictionary=None)
                f.write(json.dumps(clean(r))+'\n');f.flush();done[c['rollout_uid']]=r;reused+=1;cache_writer.append_journal([r],EXP)
            else:
                assert tuple(c[k]for k in ('state_uid','eta_uid','controller_uid','seed_key'))in missing,('cache hit needs result export',c['rollout_uid'])
                todo.append(c)
        print(stage,'requests',len(cases),'new',len(todo),'prior reused',reused,'local',len(cases)-len(todo)-reused,flush=True)
        with ProcessPoolExecutor(max_workers=min(workers or len(AVAILABLE_CPUS),len(AVAILABLE_CPUS)),mp_context=mp.get_context('spawn'),initializer=init_worker)as pool:
            for j,fu in enumerate(as_completed([pool.submit(work,c)for c in todo]),1):
                r=fu.result();f.write(json.dumps(clean(r))+'\n');f.flush();done[r['rollout_uid']]=r
                if j%20==0 or j==len(todo):print(stage,j,'/',len(todo),'elapsed',round(time.monotonic()-t0,1),flush=True)
    subprocess.run([sys.executable,'-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,check=True,stdout=(OUT/f'{stage}_merge.log').open('w'))
    dump(OUT/f'{stage}_execution.json',dict(requests=len(cases),new=len(todo),prior_reused=reused,elapsed=time.monotonic()-t0,complete=True))
    with db.connect(True)as con:
        for c in cases:
            q=con.execute('SELECT success,conflict_quarantined FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            assert q and q['success']==done[c['rollout_uid']]['success']and not q['conflict_quarantined']
    return [done[c['rollout_uid']]for c in cases]
