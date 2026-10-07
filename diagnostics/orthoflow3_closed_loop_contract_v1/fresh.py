"""One small post-repair confirmation, frozen after development; no adaptation."""
import argparse,json
import numpy as np
import jax
from diagnostics.orthoflow3_closed_loop_contract_v1 import repair
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,phase_b as b,evaluate_b as ev,rotation_confirmation_b as rot
from new_benchmark_common.basin_dataset_v1 import _environment_descriptor,CONDITIONING_FLOW_ROOT
from new_benchmark_common.safety_eta3 import state_token
from shared_rollout_db.src.rollout_db import uid,connect,canonical
OUT=repair.OUT/'confirmation'

def dump(n,x):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def freeze():
    if (OUT/'states.json').exists():raise RuntimeError('Already frozen')
    assert (repair.OUT/'development/results.json').exists()
    model=ev.load_models();b.OUT=repair.OLD_OUT;b.DATA=repair.OLD_DATA
    oldmodel=ev.load_models();repair.install()
    records=[];props=[]
    for sc in ('four_way_intersection','ring_exchange'):
        r=rot.Runtime(sc,[]);descriptor=_environment_descriptor(r)
        for i in range(2):
            draw=27_000_000+i
            if sc=='four_way_intersection':pos,vel,_=rot.four_sample(r.config,'development',draw);goals=r.make_env().goals
            else:
                initial=rot.ring_sample('development',draw,r.config);pos,vel,goals=initial.positions,initial.velocities,initial.goals
            anchor=uid('contract_confirmation',{'scenario':sc,'draw':draw})
            physical={'positions':pos.tolist(),'velocities':vel.tolist(),'goals':goals.tolist(),
                      'timestep':0,'normalized_episode_time':0.,'rng_anchor':anchor}
            ch=a.digest(physical);sid=uid('state',{'scenario':r.scenario_uid,'content':ch})
            s={'uid':sid,'alias':f'contract_confirm_{sc}_{i}','physical':physical,'content_hash':ch,'index':i,
               'provenance':{'fresh_confirmation':True,'draw':draw},'metadata':{}}
            env=r.make_env();r.reset(env,s)
            key=jax.random.fold_in(jax.random.PRNGKey(CONDITIONING_FLOW_ROOT),state_token(anchor))
            cond={'flat':np.r_[r.observation(env).ravel(),r.flow_world(env,key).ravel()].tolist()}
            row={'state_uid':sid,'scenario':sc,'structured_state':physical,'environment_descriptor':descriptor,'conditioning':cond}
            p=ev.proposals(repair.transformed(row),model)
            b.OUT=repair.OLD_OUT;b.DATA=repair.OLD_DATA
            old=ev.proposals(repair.ORIGINAL_TRANSFORM(row),oldmodel);repair.install()
            props.append({'state_uid':sid,'scenario':sc,**p,'old_eta':old['etas'][old['selected_index']]})
            records.append({'scenario':sc,**s})
    dump('states.json',records);dump('proposals.json',{'states':props,'frozen_before_outcomes':True,
         'generator':a.load(b.OUT/'generator_frozen.json'),'critic':a.load(b.OUT/'critic_frozen.json'),'no_adaptation':True})
    return {'states':len(records)}

def runtimes():
    a.bd.OUT=OUT/'registry';a.bd.WORK=OUT
    for sc in ('four_way_intersection','ring_exchange'):
        r=rot.Runtime(sc,[s for s in a.load(OUT/'states.json') if s['scenario']==sc]);r.output=OUT/sc;r.output.mkdir(parents=True,exist_ok=True)
        path=OUT/'execution';r.experiment_uid=uid('exp',{'path':str(path)})
        with connect() as con:
            con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)',
                (r.experiment_uid,'contract_fresh_confirmation',str(path),a.sha(OUT/'proposals.json'),a.sha(__file__),canonical({'fresh':True})))
            con.commit()
        yield sc,r

def jobs(r):
    by={p['state_uid']:p for p in a.load(OUT/'proposals.json')['states']}
    return [{'state':s,'eta':eta,'chain':'orthoflow3','seeds':list(a.SEEDS),'index':i}
            for s in r.states for i,eta in enumerate(by[s['uid']]['etas']+[[0,0,0],by[s['uid']]['old_eta']])]

def collect():
    truth={};rows=[]
    for sc,r in runtimes():
        for j in jobs(r):
            valid,num,missing=a.evidence(r,j)
            if missing:raise RuntimeError(missing)
            s=sum(x['success'] for x in valid.values());f=len(valid)-s
            e={'state_uid':j['state']['uid'],'scenario':sc,'index':j['index'],'successes':s,'failures':f,'numerical':len(num),
               'robust':True if s>=15 else False if f>=2 else None,'Q_lower':s/16,'Q_upper':(16-f)/16,
               'collision':sum(x['collision'] for x in valid.values())}
            truth[e['state_uid'],e['index']]=e;rows.append(e)
    states=[]
    for p in a.load(OUT/'proposals.json')['states']:
        sid=p['state_uid'];sel=truth[sid,p['selected_index']];zero=truth[sid,17];old=truth[sid,18]
        states.append({'scenario':p['scenario'],'state_uid':sid,'oracle':any(truth[sid,i]['robust'] is True for i in range(17)),
            'selected':sel['robust'],'old_selected':old['robust'],'B0':zero['robust'],
            'rescue':zero['robust'] is False and sel['robust'] is True,'break':zero['robust'] is True and sel['robust'] is False,
            'exploitation':p['scores'][p['selected_index']]>=.9375 and sel['Q_upper']<.5})
    summary={sc:{k:sum(s[k] is True for s in states if s['scenario']==sc) for k in ['oracle','selected','old_selected','B0','rescue','break','exploitation']} for sc in ('four_way_intersection','ring_exchange')}
    dump('results.json',{'summary':summary,'states':states,'evidence':rows,'no_adaptation':True});return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','cache','run','collect']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);x=p.parse_args()
    repair.install();a.bd.OUT=OUT/'registry';a.bd.WORK=OUT;ev.OUT=OUT;ev.runtimes=runtimes;ev.jobs=jobs
    print(json.dumps({'freeze':freeze,'cache':ev.cache,'run':lambda:ev.run(x.shard,x.shards),'collect':collect}[x.stage](),indent=2))
