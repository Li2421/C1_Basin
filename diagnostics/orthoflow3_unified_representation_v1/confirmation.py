"""One fresh cohort after learning freeze, with active-rotation controls."""
import argparse,json
from types import SimpleNamespace
import numpy as np
import jax
from . import build,train,analyze,evaluate
from diagnostics.orthoflow3_closed_loop_contract_v1 import repair
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,phase_b as b,evaluate_b as ev,rotation_confirmation_b as rot
from new_benchmark_common.basin_dataset_v1 import _environment_descriptor,CONDITIONING_FLOW_ROOT
from new_benchmark_common.safety_eta3 import state_token
from shared_rollout_db.src.rollout_db import uid,connect,canonical
OUT=build.OUT/'confirmation'

def dump(n,x):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

class DoubleRuntime(a.frozen.DoubleFrozenRuntime):
    def _states(self):return []
    def _episode(self,state):
        cfg=a.frozen.db_old.Config().to_dict();p=state['physical']
        ep=SimpleNamespace(initial_positions=np.asarray(p['positions']),initial_velocities=np.asarray(p['velocities']),regime=state['metadata']['regime'])
        return SimpleNamespace(config=cfg),ep

def double_runtime():
    a.frozen.OUT=OUT/'double_registry'
    source=next(iter(a.bd._double_source_metadata().values()))['dataset']
    a.frozen.DOUBLE_POOLS={'source_train_pool_for_checkpoint_schema':source}
    return DoubleRuntime()

def register(r):
    r.output=OUT/r.name;r.output.mkdir(parents=True,exist_ok=True)
    path=OUT/'execution';r.experiment_uid=uid('exp',{'path':str(path)})
    with connect() as con:
        con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)',
            (r.experiment_uid,'unified_representation_confirmation',str(path),a.sha(OUT/'proposals.json'),a.sha(__file__),canonical({'fresh':True})))
        for s in r.states:
            con.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                (s['uid'],r.scenario_uid,'unified_fresh_confirmation',s['content_hash'],canonical(s['physical']),canonical({'goals':s['physical']['goals']}),canonical(s['provenance']),'CONTENT_EXACT'))
            con.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(r.scenario_uid,s['alias'],s['uid'],r.experiment_uid))
        con.commit()

def freeze():
    if (OUT/'states.json').exists():raise RuntimeError('Already frozen')
    dev=a.load(evaluate.OUT/'results.json')['summary']
    for sc in a.SCENARIOS:
        assert dev[sc]['unified']['oracle']>=dev[sc]['old']['oracle'],('oracle learning gate',sc)
        assert dev[sc]['unified']['selected']>=dev[sc]['old']['selected'],('critic learning gate',sc)
    models=train.load_models();repair.install();oldmodel=ev.load_models();records=[];pp=[]
    def add(r,s,flow,descriptor):
        env=r.make_env() if r.name!='double_bottleneck' else None
        if env is not None:r.reset(env,s);obs=r.observation(env)
        else:
            dataset,ep=r._episode(s);env=a.frozen.db_old._initialize_env(a.frozen.db_old.Config(**dataset.config),ep);obs=env.observation()
        row={'state_uid':s['uid'],'scenario':r.name,'structured_state':s['physical'],'environment_descriptor':descriptor,
             'conditioning':{'flat':np.r_[obs.ravel(),flow.ravel()].tolist()}}
        p=analyze.proposals(build.convert(row),models,seed_identity=s['physical'].get('rng_anchor',s['uid']))
        old=ev.proposals(repair.transformed(row),oldmodel,seed_identity=s['physical'].get('rng_anchor',s['uid']))
        pp.append({'scenario':r.name,'state_uid':s['uid'],'rotation':s['metadata'].get('rotation',0),
            'anchor':s['physical'].get('rng_anchor',s['uid']),**p,'old_eta':old['etas'][old['selected_index']]})
        records.append({'scenario':r.name,**s})
    # Three outcome-blind legitimate Double initial draws, one per original regime.
    from diagnostics.double_bottleneck_initial_state_coverage.tools.generate_pools import _sample_spec,_seed,INITIAL_REGIMES
    r=double_runtime();cfg=a.frozen.db_old.Config()
    for i,regime in enumerate(INITIAL_REGIMES):
        rng=np.random.default_rng(_seed('unified_rep_confirmation_v1',regime));spec=_sample_spec('unified_rep_confirmation_v1',regime,0,rng)
        ep=SimpleNamespace(initial_positions=spec.positions,initial_velocities=spec.initial_velocities,regime=regime)
        env=a.frozen.db_old._initialize_env(cfg,ep)
        key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(731581),i),0)
        flow=np.asarray(r.policy.sample_actions(env.observation()[None],key)[0],float)
        physical={'positions':env.positions.tolist(),'velocities':spec.initial_velocities.tolist(),'goals':env.goals.tolist(),'timestep':0,'normalized_episode_time':0.}
        physical['committed_current_flow']=flow.tolist()
        ch=a.digest(physical);sid=uid('state',{'scenario':r.scenario_uid,'content':ch})
        s={'uid':sid,'alias':f'UNIFIED_CONFIRM_DB_{i}','content_hash':ch,'physical':physical,'index':i,
           'metadata':{'historical_seed':731581,'rollout_id':i,'regime':regime,'family_id':spec.family_id},
           'provenance':{'fresh_confirmation':True,'distribution':'original declared initial-state sampler','regime':regime}}
        add(r,s,flow,cfg.to_dict())
    for sc in ('four_way_intersection','ring_exchange'):
        r=rot.Runtime(sc,[]);descriptor=_environment_descriptor(r)
        for i in range(2):
            draw=33_000_000+i
            if sc=='four_way_intersection':pos,vel,_=rot.four_sample(r.config,'development',draw);goals=r.make_env().goals
            else:
                init=rot.ring_sample('development',draw,r.config);pos,vel,goals=init.positions,init.velocities,init.goals
            anchor=uid('unified_confirmation_anchor',{'scenario':sc,'draw':draw})
            key=jax.random.fold_in(jax.random.PRNGKey(CONDITIONING_FLOW_ROOT),state_token(anchor))
            for k in range(4):
                physical={n:rot.cc.rotate(v,k).tolist() for n,v in [('positions',pos),('velocities',vel),('goals',goals)]}
                physical.update(timestep=0,normalized_episode_time=0.,rng_anchor=anchor)
                ch=a.digest(physical);sid=uid('state',{'scenario':r.scenario_uid,'content':ch})
                s={'uid':sid,'alias':f'UNIFIED_CONFIRM_{sc}_{i}_{90*k}','content_hash':ch,'physical':physical,'index':i,
                    'metadata':{'rotation':90*k,'draw':draw},'provenance':{'fresh_confirmation':True,'draw':draw,'rotation':90*k}}
                env=r.make_env();r.reset(env,s);flow=r.flow_world(env,key);add(r,s,flow,descriptor)
    dump('states.json',records);dump('proposals.json',{'states':pp,'frozen_before_outcomes':True,
        'no_adaptation':True,'generator':a.load(build.OUT/'generator_frozen.json'),'critic':a.load(build.OUT/'critic_frozen.json'),
        'Four_active_rotations':'non-equivalent upstream-policy controls, not required value invariance',
        'Double_sampler':'same declared original global sampler; no expert/outcome rejection'})
    return {'states':len(records)}

def runtimes():
    states=a.load(OUT/'states.json');a.bd.OUT=OUT/'registry';a.bd.WORK=OUT
    r=double_runtime();r.states=[s for s in states if s['scenario']==r.name];register(r);yield r.name,r
    for sc in ('four_way_intersection','ring_exchange'):
        r=rot.Runtime(sc,[s for s in states if s['scenario']==sc]);register(r);yield sc,r

def jobs(r):
    by={p['state_uid']:p for p in a.load(OUT/'proposals.json')['states']}
    return [{'state':s,'eta':eta,'chain':'orthoflow3','seeds':list(a.SEEDS),'index':i}
        for s in r.states for i,eta in enumerate(by[s['uid']]['etas']+[[0,0,0],by[s['uid']]['old_eta']])]

def collect():
    truth={};evidence=[]
    for sc,r in runtimes():
        for j in jobs(r):
            valid,num,missing=a.evidence(r,j)
            if missing:raise RuntimeError(missing)
            ss=sum(x['success'] for x in valid.values());ff=len(valid)-ss
            e={'scenario':sc,'state_uid':j['state']['uid'],'index':j['index'],'successes':ss,'failures':ff,'numerical':len(num),
               'robust':True if ss>=15 else False if ff>=2 else None,'Q_lower':ss/16,'Q_upper':(16-ff)/16,
               'collisions':sum(x['collision'] for x in valid.values()),'outcomes':{str(k):x['outcome'] for k,x in valid.items()}}
            truth[e['state_uid'],e['index']]=e;evidence.append(e)
    states=[]
    for p in a.load(OUT/'proposals.json')['states']:
        sid=p['state_uid'];s=truth[sid,p['selected_index']];z=truth[sid,17];old=truth[sid,18]
        states.append({'scenario':p['scenario'],'state_uid':sid,'rotation':p['rotation'],'anchor':p['anchor'],
            'oracle':any(truth[sid,i]['robust'] is True for i in range(17)),'selected':s['robust'],'old_selected':old['robust'],'B0':z['robust'],
            'rescue':z['robust'] is False and s['robust'] is True,'break':z['robust'] is True and s['robust'] is False,
            'exploitation':p['scores'][p['selected_index']]>=.9375 and s['Q_upper']<.5,
            'selected_index':p['selected_index'],'selected_outcomes':s['outcomes'],'selected_eta':p['etas'][p['selected_index']]})
    summary={}
    for sc in a.SCENARIOS:
        summary[sc]={}
        for deg in sorted({s['rotation'] for s in states if s['scenario']==sc}):
            rr=[s for s in states if s['scenario']==sc and s['rotation']==deg]
            summary[sc][str(deg)]={'states':len(rr),**{k:sum(s[k] is True for s in rr) for k in ['oracle','selected','old_selected','B0','rescue','break','exploitation']}}
    dump('results.json',{'summary':summary,'states':states,'evidence':evidence,'no_adaptation':True});return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','cache','run','collect']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);x=p.parse_args()
    a.bd.OUT=OUT/'registry';a.bd.WORK=OUT;ev.OUT=OUT;ev.runtimes=runtimes;ev.jobs=jobs
    print(json.dumps({'freeze':freeze,'cache':ev.cache,'run':lambda:ev.run(x.shard,x.shards),'collect':collect}[x.stage](),indent=2))
