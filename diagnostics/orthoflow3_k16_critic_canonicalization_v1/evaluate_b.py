"""Fixed train/dev comparison for canonical conditioning, DB-backed Q16."""
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
import numpy as np
import jax.numpy as jnp
from flax import serialization
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,phase_b as b,train_phase_a as ta
from new_benchmark_common.safety_eta3 import DatabaseSink,request,preflight
from shared_rollout_db.src.rollout_db import canonical,connect,uid
OUT=b.OUT/'development'


def dump(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')


def load_models():
    norm=a.load(b.OUT/'normalization.json');dims={s:len(norm['scenarios'][s]['h_mean']) for s in a.SCENARIOS};cdim=len(norm['environment_keys'])
    gm,cm=a.learn.Generator(),a.learn.Critic()
    gp=serialization.from_bytes(a.learn.merge_initialized(gm,dims,cdim),__import__('pathlib').Path(a.load(b.OUT/'generator_frozen.json')['selected']['checkpoint']).read_bytes())
    cp=serialization.from_bytes(a.learn.merge_initialized(cm,dims,cdim,critic=True),__import__('pathlib').Path(a.load(b.OUT/'critic_frozen.json')['selected']['checkpoint']).read_bytes())
    return gm,gp,cm,cp


def proposals(row,models,seed_identity=None):
    gm,gp,cm,cp=models;sc=row['scenario'];h,c=b.inputs(row)
    raw=gm.apply(gp,jnp.asarray(h[None]),jnp.asarray(c[None]),method=getattr(gm,a.learn.METHOD[sc]))[0]
    rng=np.random.default_rng(a.learn.stable_int('generator-v1-proposals',41,sc,seed_identity or row['state_uid']))
    noise=jnp.asarray(rng.standard_normal((16,3)),jnp.float32)
    etas=np.asarray([a.learn.eta_mean(raw),*a.learn.eta_from_noise(raw[None].repeat(16,0),noise)],float)
    logits=np.asarray(cm.apply(cp,jnp.asarray(h[None].repeat(17,0)),jnp.asarray(c[None].repeat(17,0)),
                    jnp.asarray((etas-a.learn.CENTER)/a.learn.RADIUS),method=getattr(cm,a.learn.METHOD[sc])))
    return {'etas':etas.tolist(),'logits':logits.tolist(),'scores':(1/(1+np.exp(-logits))).tolist(),
            'selected_index':int(np.argmax(logits)),'proposal_hash':a.digest(etas.tolist())}


def freeze():
    if (OUT/'proposals.json').exists():raise RuntimeError('Already frozen')
    selected=[];model=load_models();oldnorm=a.load(a.frozen.NORMALIZATION)
    _,_,oldcm,oldcp=a.fresh.load_models(oldnorm)
    oldcp=serialization.from_bytes(oldcp,__import__('pathlib').Path(a.load(a.OUT/'phase_a/critic_frozen.json')['checkpoint']).read_bytes())
    oldrows={r['state_uid']:r for r in a.states()};oldprops={r['state_uid']:r for r in ta.all_proposals()}
    for sc in a.SCENARIOS:
        cohort=sorted([r for r in b.rows() if r['scenario']==sc and r['split']=='validation'],key=lambda r:a.digest('phase-b-dev|'+r['state_uid']))[:12]
        for row in cohort:
            sid=row['state_uid'];h,c=a.inputs(oldrows[sid],oldnorm);old=oldprops[sid]
            logits=np.asarray(oldcm.apply(oldcp,jnp.asarray(h[None].repeat(17,0)),jnp.asarray(c[None].repeat(17,0)),
                       jnp.asarray((np.asarray(old['etas'])-a.learn.CENTER)/a.learn.RADIUS),method=getattr(oldcm,a.learn.METHOD[sc])))
            selected.append({'state_uid':sid,'state_id':row['state_id'],'scenario':sc,'split':row['split'],
                'parent_episode_id':row['parent_episode_id'],**proposals(row,model),
                'old_selected_index':int(np.argmax(logits)),'old_proposal_hash':old['proposal_hash'],
                'old_scores':(1/(1+np.exp(-logits))).tolist()})
    d={'frozen_before_new_outcomes':True,'K_stochastic':16,'states':selected,'model_hashes':{
        'generator':a.load(b.OUT/'generator_frozen.json')['selected']['sha256'],
        'critic':a.load(b.OUT/'critic_frozen.json')['selected']['sha256'],'conditioning':a.sha(b.DATA/'states.parquet')}}
    dump('proposals.json',d);return {'states':len(selected),'requested_seed_slots':len(selected)*18*16}


def register(r):
    r.output=OUT/r.name;r.output.mkdir(parents=True,exist_ok=True)
    provenance=OUT/'execution';r.experiment_uid=uid('exp',{'path':str(provenance)})
    with connect() as con:
        con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)',
                    (r.experiment_uid,'phase_B_development',str(provenance),a.sha(OUT/'proposals.json'),a.sha(__file__),canonical({'train_dev_only':True})))
        assert con.execute('SELECT 1 FROM experiment WHERE experiment_uid=?',(r.experiment_uid,)).fetchone()
        con.commit()


def runtimes():
    wanted={r['state_uid'] for r in a.load(OUT/'proposals.json')['states']}
    for sc,r in a.runtimes():
        r.states=[s for s in r.states if s['uid'] in wanted]
        if r.states:register(r);yield sc,r


def jobs(r):
    by={p['state_uid']:p for p in a.load(OUT/'proposals.json')['states']}
    return [{'state':s,'eta':eta,'chain':'orthoflow3','seeds':list(a.SEEDS),'index':i if i<17 else None}
            for s in r.states for i,eta in enumerate(by[s['uid']]['etas']+[[0,0,0]])]


def run(shard,shards):
    totals=Counter();index=0;seen=set()
    for sc,r in runtimes():
        chosen=[]
        for job in jobs(r):
            identity=(job['state']['uid'],a.eta_identity(job['eta'])[0],r.controllers['orthoflow3']['uid'])
            if identity in seen:continue
            seen.add(identity)
            if index%shards==shard:chosen.append(job)
            index+=1
        if not chosen:continue
        stem=f'cache/{sc}_{r.controllers["orthoflow3"]["uid"][-8:]}_{shard}of{shards}'
        dump(stem+'.json',{'requests':[request(r,j['state'],j['eta'],'orthoflow3',a.SEEDS) for j in chosen]})
        dump(stem+'_preflight.json',preflight(OUT/(stem+'.json')))
        sink=DatabaseSink(r,f'phase_b_{shard}of{shards}')
        try:
            for j in chosen:
                found,num,missing=a.evidence(r,j);totals['requested']+=16;totals['reused']+=len(found);totals['existing_numerical']+=len(num)
                for seed in missing:
                    for attempt in range(4):
                        row=r.rollout(j['state'],np.asarray(j['eta']),seed,'orthoflow3')
                        row.update(proposal_index=j['index'],execution_attempt=attempt,
                                   phase='B_rotation_confirmation' if OUT.name=='rotation_confirmation' else 'B_development')
                        sink.insert(row);totals['physical_attempts']+=1
                        if not row['numerical_failure']:break
                    totals['completed_seeds']+=1;totals['numerical_unresolved']+=bool(row['numerical_failure'])
                    if totals['physical_attempts']%50==0:print(json.dumps({'shard':shard,**totals}),flush=True)
        finally:sink.finalize()
    dump(f'run_{shard}of{shards}.json',dict(totals));return dict(totals)


def cache():
    totals=Counter();unique=set();missing_unique=set();reused_unique=set()
    for sc,r in runtimes():
        jj=jobs(r);stem=f'cache/initial_{sc}_{r.controllers["orthoflow3"]["uid"][-8:]}'
        dump(stem+'.json',{'requests':[request(r,j['state'],j['eta'],'orthoflow3',a.SEEDS) for j in jj]})
        dump(stem+'_preflight.json',preflight(OUT/(stem+'.json')))
        for j in jj:
            valid,num,missing=a.evidence(r,j)
            totals['requested']+=16;totals['exact_reused_seeds']+=len(valid)
            totals['numerical_uncertified_existing']+=len(num);totals['truly_missing']+=len(missing)
            totals['partial_tuple_reuse']+=bool(valid) and bool(missing)
            prefix=(j['state']['uid'],a.eta_identity(j['eta'])[0],r.controllers['orthoflow3']['uid'])
            unique.update((*prefix,s) for s in a.SEEDS)
            missing_unique.update((*prefix,s) for s in missing)
            reused_unique.update((*prefix,s) for s in valid)
    totals['aggregate_only_reuse']=0
    totals['requested_unique']=len(unique);totals['truly_missing_unique']=len(missing_unique);totals['exact_reused_unique']=len(reused_unique)
    dump('cache_preflight.json',dict(totals));return dict(totals)


def collect():
    truth={};missing=[]
    for sc,r in runtimes():
        for j in jobs(r):
            valid,num,absent=a.evidence(r,j);s=sum(bool(x['success']) for x in valid.values());f=len(valid)-s
            e={'scenario':sc,'state_uid':j['state']['uid'],'candidate':j['index'],'successes':s,'failures':f,
               'numerical_seeds':list(num),'robust':True if s>=15 else False if f>=2 else None,
               'Q16':s/16 if len(valid)==16 else None,'Q_lower':s/16,'Q_upper':(16-f)/16,
               'collisions':sum(bool(x['collision']) for x in valid.values()),'rollout_uids':[x['rollout_uid'] for x in valid.values()]}
            truth[(e['state_uid'],e['candidate'])]=e
            if absent:missing.append([e['state_uid'],e['candidate'],absent])
    dump('seed_evidence.json',{'rows':list(truth.values()),'missing':missing})
    if missing:raise RuntimeError(f'Missing {len(missing)} tuples')
    old={(x['state_uid'],x['proposal_index']):x for x in ta.all_evidence()};result=[]
    for p in a.load(OUT/'proposals.json')['states']:
        sid=p['state_uid'];ee=[truth[(sid,i)] for i in range(17)];chosen=ee[p['selected_index']];zero=truth[(sid,None)];oe=old[(sid,p['old_selected_index'])]
        oracle=True if any(x['robust'] is True for x in ee) else None if any(x['robust'] is None for x in ee) else False
        pairs=correct=0
        for j in range(17):
            for k in range(j):
                if ee[j]['Q16'] is None or ee[k]['Q16'] is None:continue
                diff=ee[j]['Q16']-ee[k]['Q16']
                if abs(diff)<1/16:continue
                pairs+=1;correct+=(p['logits'][j]-p['logits'][k])*diff>0
        result.append({'scenario':p['scenario'],'state_uid':sid,'B0_robust':zero['robust'],'oracle_robust':oracle,
            'old_oracle_robust':any(old[(sid,j)]['robust'] is True for j in range(17)),
            'selected_robust':chosen['robust'],'old_selected_robust':oe['robust'],'selected_index':p['selected_index'],
            'robust_proposal_count':sum(x['robust'] is True for x in ee),'pairwise_correct':int(correct),'pairwise_pairs':pairs,
            'rescue':zero['robust'] is False and chosen['robust'] is True,'break':zero['robust'] is True and chosen['robust'] is False,
            'miss':oracle is True and chosen['robust'] is False,'unresolved':chosen['robust'] is None,
            'exploitation':p['scores'][p['selected_index']]>=.9375 and chosen['Q_upper']<.5,
            'regret_lower':max(0.,max(x['Q_lower'] for x in ee)-chosen['Q_upper']),
            'regret_upper':max(x['Q_upper'] for x in ee)-chosen['Q_lower']})
    summary={}
    for sc in a.SCENARIOS:
        rr=[x for x in result if x['scenario']==sc]
        summary[sc]={'states':len(rr),**{k:sum(x[k] is True for x in rr) for k in ['B0_robust','oracle_robust','old_oracle_robust','selected_robust','old_selected_robust','rescue','break','miss','unresolved','exploitation']},
                     'mean_robust_proposals':float(np.mean([x['robust_proposal_count'] for x in rr])),
                     'pairwise_accuracy':sum(x['pairwise_correct'] for x in rr)/max(1,sum(x['pairwise_pairs'] for x in rr)),
                     'regret_lower':float(np.mean([x['regret_lower'] for x in rr])),'regret_upper':float(np.mean([x['regret_upper'] for x in rr]))}
    dump('results.json',{'summary':summary,'states':result});return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','cache','run','collect']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);x=p.parse_args()
    print(json.dumps({'freeze':freeze,'cache':cache,'run':lambda:run(x.shard,x.shards),'collect':collect}[x.stage](),indent=2))
