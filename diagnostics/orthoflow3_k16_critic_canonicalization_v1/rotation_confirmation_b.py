"""Frozen small new physical-rotation cohort; no adaptation to outcomes."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,phase_b as b,evaluate_b as ev,canonical_conditioning as cc
from new_benchmark_common.basin_dataset_v1 import TrainingRuntime,_environment_descriptor,CONDITIONING_FLOW_ROOT
from new_benchmark_common.safety_eta3 import state_token,FUTURE_ROOT
from shared_rollout_db.src.rollout_db import uid,connect,canonical
from four_way_intersection.scenario import sample_initial_state as four_sample
from ring_exchange.environment import sample_initial_state as ring_sample
OUT=b.OUT/'rotation_confirmation'
a.bd.OUT=OUT/'registry'
a.bd.WORK=OUT


def dump(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')


class Runtime(TrainingRuntime):
    def _training_controller(self,chain,parent):
        d=super()._training_controller(chain,parent);p=d['payload']
        p['conditioning']='phase_B_new_rotation_snapshot_v1'
        p['rng']={'root':FUTURE_ROOT,'derivation':'state token from physical snapshot rng_anchor;future_index;step'}
        p['goal_restore']='restore explicit physical goals for Four and Ring'
        return {'payload':p,'uid':uid('ctl',p)}

    def reset(self,env,state):
        super().reset(env,state)
        env.goals=np.asarray(state['physical']['goals'],float).copy()

    def rollout(self,state,eta,future_index,chain,*,save_trace=False):
        proxy={**state,'uid':state['physical']['rng_anchor']}
        row=super().rollout(proxy,eta,future_index,chain,save_trace=future_index==0)
        row.update(state_uid=state['uid'],state_id=state['alias'],rng_anchor=proxy['uid'])
        return row


def freeze():
    if (OUT/'proposals.json').exists():raise RuntimeError('Already frozen')
    assert (b.OUT/'development/results.json').exists(), 'Complete development before fresh confirmation'
    model=ev.load_models();oldnorm=a.load(a.frozen.NORMALIZATION);gm,gp,cm,cp=a.fresh.load_models(oldnorm)
    cp=serialization.from_bytes(cp,Path(a.load(a.OUT/'phase_a/critic_frozen.json')['checkpoint']).read_bytes())
    records=[];props=[]
    for sc in ('four_way_intersection','ring_exchange'):
        r=Runtime(sc,[]);descriptor=_environment_descriptor(r)
        for i in range(4):
            draw=12_000_000+i
            if sc=='four_way_intersection':
                pos,vel,_=four_sample(r.config,'development',draw);goals=r.make_env().goals
            else:
                initial=ring_sample('development',draw,r.config);pos,vel,goals=initial.positions,initial.velocities,initial.goals
            anchor=uid('rotation_anchor',{'scenario':sc,'draw':draw,'physical':[pos.tolist(),vel.tolist(),goals.tolist()]})
            key=jax.random.fold_in(jax.random.PRNGKey(CONDITIONING_FLOW_ROOT),state_token(anchor))
            for k in range(4):
                physical={n:cc.rotate(x,k).tolist() for n,x in [('positions',pos),('velocities',vel),('goals',goals)]}
                physical.update(timestep=0,normalized_episode_time=0.,rng_anchor=anchor)
                ch=a.digest(physical);sid=uid('state',{'scenario':r.scenario_uid,'content':ch})
                s={'uid':sid,'alias':f'B_ROT_{sc}_{i}_{90*k}','physical':physical,'content_hash':ch,'index':i,
                   'provenance':{'new_confirmation':True,'rotation':90*k,'draw':draw,'rng_anchor':anchor},'metadata':{'rotation':90*k,'draw':draw}}
                env=r.make_env();r.reset(env,s);flow=r.flow_world(env,key)
                cond={'flat':np.r_[r.observation(env).ravel(),flow.ravel()].tolist()}
                row={'state_uid':sid,'scenario':sc,'conditioning':json.dumps(cond),'structured_state':json.dumps(physical),'environment_descriptor':json.dumps(descriptor)}
                p=ev.proposals(b.transformed(row),model,seed_identity=anchor)
                h,c=a.inputs(row,oldnorm);raw=gm.apply(gp,jnp.asarray(h[None]),jnp.asarray(c[None]),method=getattr(gm,a.learn.METHOD[sc]))[0]
                noise=jnp.asarray(np.random.default_rng(a.learn.stable_int('generator-v1-proposals',41,sc,anchor)).standard_normal((16,3)),jnp.float32)
                oe=np.asarray([a.learn.eta_mean(raw),*a.learn.eta_from_noise(raw[None].repeat(16,0),noise)])
                logits=np.asarray(cm.apply(cp,jnp.asarray(h[None].repeat(17,0)),jnp.asarray(c[None].repeat(17,0)),
                                  jnp.asarray((oe-a.learn.CENTER)/a.learn.RADIUS),method=getattr(cm,a.learn.METHOD[sc])))
                records.append({'scenario':sc,**s});props.append({'scenario':sc,'state_uid':sid,'rotation':90*k,'anchor':anchor,**p,
                    'old_eta':oe[int(np.argmax(logits))].tolist(),'old_selected_index':int(np.argmax(logits))})
    dump('states.json',records);dump('proposals.json',{'states':props,'frozen_before_outcomes':True,'new_base_states_per_scenario':4,
        'models':{'generator':a.load(b.OUT/'generator_frozen.json'),'critic':a.load(b.OUT/'critic_frozen.json')},
        'no_adaptation_after_confirmation':True})
    return {'physical_states':len(records),'base_states':8}


def runtimes():
    rows=a.load(OUT/'states.json')
    for sc in ('four_way_intersection','ring_exchange'):
        r=Runtime(sc,[s for s in rows if s['scenario']==sc]);r.output=OUT/sc;r.output.mkdir(parents=True,exist_ok=True)
        path=OUT/'execution';r.experiment_uid=uid('exp',{'path':str(path)})
        with connect() as con:
            con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)',
              (r.experiment_uid,'phase_B_rotation_confirmation',str(path),a.sha(OUT/'proposals.json'),a.sha(__file__),canonical({'fresh':True})))
            con.commit()
        yield sc,r


def jobs(r):
    by={p['state_uid']:p for p in a.load(OUT/'proposals.json')['states']};jj=[]
    for s in r.states:
        p=by[s['uid']]
        candidates=[('selected',p['etas'][p['selected_index']]),('old',p['old_eta']),('B0',[0,0,0])]
        if p['rotation']==0:candidates.extend((f'proposal_{i}',eta) for i,eta in enumerate(p['etas']))
        for index,eta in candidates:jj.append({'state':s,'eta':eta,'chain':'orthoflow3','seeds':list(a.SEEDS),'index':index})
    return jj


def collect():
    truth={};details=[]
    for sc,r in runtimes():
        for j in jobs(r):
            valid,num,missing=a.evidence(r,j)
            if missing:raise RuntimeError(('missing',j['state']['uid'],missing))
            ss=sum(x['success'] for x in valid.values());ff=len(valid)-ss
            e={'scenario':sc,'state_uid':j['state']['uid'],'method':j['index'],'successes':ss,'failures':ff,
                'numerical':len(num),'robust':True if ss>=15 else False if ff>=2 else None,'collision':sum(x['collision'] for x in valid.values()),
                'mean_length':float(np.mean([x['episode_length'] for x in valid.values()])) if valid else None,
                'Q_lower':ss/16,'Q_upper':(16-ff)/16,'outcomes':{str(k):x['outcome'] for k,x in valid.items()}}
            truth[(e['state_uid'],e['method'])]=e;details.append(e)
    pp=a.load(OUT/'proposals.json')['states'];summary={}
    for sc in ('four_way_intersection','ring_exchange'):
        summary[sc]={}
        for deg in (0,90,180,270):
            rr=[p for p in pp if p['scenario']==sc and p['rotation']==deg];item={}
            for method in ('B0','old','selected'):
                ee=[truth[(p['state_uid'],method)] for p in rr]
                item[method]={'robust':sum(e['robust'] is True for e in ee),'unresolved':sum(e['robust'] is None for e in ee),
                    'numerical_seeds':sum(e['numerical'] for e in ee),'collisions':sum(e['collision'] for e in ee)}
            if deg==0:item['oracle_robust']=sum(any(truth[(p['state_uid'],f'proposal_{i}')]['robust'] is True for i in range(17)) for p in rr)
            summary[sc][str(deg)]=item
    traces={}
    for path in OUT.glob('*/raw/*.jsonl'):
        for line in path.open():
            x=json.loads(line)
            if x['future_index']==0 and not x['numerical_failure'] and 'trace' in x:
                traces[(x['state_uid'],a.eta_identity(x['eta'])[0])]=x
    paired=[]
    base={p['anchor']:p for p in pp if p['rotation']==0}
    for p in pp:
        if p['rotation']==0:continue
        original=base[p['anchor']]
        for method in ('old','selected'):
            eta0=original['old_eta'] if method=='old' else original['etas'][original['selected_index']]
            eta1=p['old_eta'] if method=='old' else p['etas'][p['selected_index']]
            x0=traces.get((original['state_uid'],a.eta_identity(eta0)[0]));x1=traces.get((p['state_uid'],a.eta_identity(eta1)[0]))
            rmse=None
            if x0 and x1:
                p0=np.asarray(x0['trace']['positions']);p1=cc.rotate(x1['trace']['positions'],(-p['rotation']//90)%4)
                n=min(len(p0),len(p1));rmse=float(np.sqrt(np.mean((p0[:n]-p1[:n])**2)))
            z0=truth[(original['state_uid'],method)];z1=truth[(p['state_uid'],method)]
            seeds=set(z0['outcomes'])&set(z1['outcomes'])
            paired.append({'scenario':p['scenario'],'anchor':p['anchor'],'rotation':p['rotation'],'method':method,
                'eta_delta':float(np.linalg.norm((np.asarray(eta0)-eta1)/a.learn.RADIUS)),
                'trajectory_rmse':rmse,'paired_certified_seeds':len(seeds),
                'outcome_agreement':sum(z0['outcomes'][s]==z1['outcomes'][s] for s in seeds)/len(seeds) if seeds else None,
                'mean_length_original':z0['mean_length'],'mean_length_transformed':z1['mean_length']})
    dump('results.json',{'summary':summary,'evidence':details,'paired_trajectories':paired,'no_adaptation':True});return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','cache','run','collect']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);x=p.parse_args()
    ev.OUT=OUT;ev.runtimes=runtimes;ev.jobs=jobs
    print(json.dumps({'freeze':freeze,'cache':ev.cache,'run':lambda:ev.run(x.shard,x.shards),'collect':collect}[x.stage](),indent=2))
