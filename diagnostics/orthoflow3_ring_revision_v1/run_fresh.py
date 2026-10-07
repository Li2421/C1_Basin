#!/usr/bin/env python3
"""Fresh untouched Q16 confirmation after Ring revision decisions are frozen."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json,sqlite3

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_generator_critic_frozen_test_v1 import run_frozen_test as old
from diagnostics.orthoflow3_generator_critic_v1.train_evaluate import (Generator,Critic,METHOD,
    CENTER,RADIUS,eta_mean,eta_from_noise,merge_initialized,stable_int,scalar_environment_fields)
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from four_way_intersection.scenario import sample_initial_state as sample_four
from ring_exchange.environment import sample_initial_state as sample_ring
from shared_rollout_db.src.rollout_db import canonical,connect,eta_identity,initialize,uid

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
EXPERIMENT='orthoflow3_ring_revision_fresh_test_v1';SEEDS=tuple(range(16));K=4
FAIR=json.loads((OUT/'fair_fixed_eta.json').read_text())
FAIR_ETA={k:v['selected']['eta'] for k,v in FAIR.items()}

def dump(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def h(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def register_fresh(runtime,scenario):
 runtime.output=OUT/scenario;runtime.output.mkdir(parents=True,exist_ok=True)
 runtime.experiment_uid=uid('exp',{'path':str(OUT.resolve()),'revision':1})
 initialize()
 with connect() as con:
  con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)',(runtime.experiment_uid,EXPERIMENT,str(OUT.resolve()),h({'K':K,'seeds':SEEDS,'no_eta0':True}),filehash(Path(__file__)),canonical({'fresh_test':True,'decisions_frozen':True})))
  for s in runtime.states:
   con.execute('''INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)''',(s['uid'],runtime.scenario_uid,'ring_revision_fresh_test',s['content_hash'],canonical(s['physical']),canonical({'goals':s['physical'].get('goals')}),canonical(s['metadata']),'CONTENT_EXACT'))
   con.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(runtime.scenario_uid,s['alias'],s['uid'],runtime.experiment_uid))
  con.commit()

def fresh_new_runtime(scenario,n):
 r=old.FrozenNewRuntime(scenario);states=[]
 if scenario=='four_way_intersection':
  for i in range(n):
   pos,vel,meta=sample_four(r.config,'test',3_000_000+i)
   physical={'positions':pos.tolist(),'velocities':vel.tolist(),'goals':None}
   ch=h(physical);states.append({'index':i,'alias':f'RREV_FOUR_TEST_{i:03d}','uid':uid('state',{'scenario':r.scenario_uid,'content':ch}),'content_hash':ch,'physical':physical,'metadata':{'distribution':'test','seed_index':3_000_000+i,'draw':meta,'untouched':True}})
 else:
  for i in range(n):
   x=sample_ring('test',4_000_000+i,r.config);physical={'positions':x.positions.tolist(),'velocities':x.velocities.tolist(),'goals':x.goals.tolist()}
   ch=h(physical);states.append({'index':i,'alias':f'RREV_RING_TEST_{i:03d}','uid':uid('state',{'scenario':r.scenario_uid,'content':ch}),'content_hash':ch,'physical':physical,'metadata':{'distribution':'test','seed_index':4_000_000+i,'untouched':True}})
 r.states=states;register_fresh(r,scenario);return r

def fresh_double_runtime():
 r=old.DoubleFrozenRuntime();d=FlowBC4ADataset(OUT/'fresh_double_pool','val',seed=0);r.datasets={'revision':d};states=[]
 for i,family in enumerate(d.family_names):
  ep=d.by_family[family][0];physical={'positions':ep.initial_positions.tolist(),'velocities':ep.initial_velocities.tolist()};ch=h(physical)
  # Frozen conditioning action uses the same current-action key as rollout.
  key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(92001),i),0);env=old.db_old._initialize_env(old.db_old.Config(**d.config),ep);raw=np.asarray(r.policy.sample_actions(env.observation()[None],key)[0],float)
  states.append({'uid':uid('state',{'scenario':r.scenario_uid,'content':ch}),'alias':f'RREV_DB_TEST_{i:03d}','index':i,'content_hash':ch,'physical':physical,'conditioning_flat':np.r_[env.observation().ravel(),raw.ravel()].tolist(),'environment_descriptor':d.config,'metadata':{'set':'revision','family_id':family,'historical_seed':92001,'rollout_id':i,'fresh_test':True}})
 r.states=states;register_fresh(r,'double_bottleneck');return r

def runtimes():return {'double_bottleneck':fresh_double_runtime(),'four_way_intersection':fresh_new_runtime('four_way_intersection',24),'ring_exchange':fresh_new_runtime('ring_exchange',60)}

def context_vector(env,norm,scenario):
 values=scalar_environment_fields(env);keys=norm['environment_keys'];c=np.zeros(len(keys)+3,np.float32)
 for j,k in enumerate(keys):c[j]=values.get(k,0.)
 c[len(keys)+old.SCENARIOS.index(scenario)]=1.;n=norm['scenarios'][scenario]
 return (c-np.asarray(n['c_mean'],np.float32))/np.asarray(n['c_std'],np.float32)

def load_models(norm):
 dims={s:len(norm['scenarios'][s]['h_mean']) for s in old.SCENARIOS};cdim=len(norm['environment_keys'])+3
 gm=Generator();gp=serialization.from_bytes(merge_initialized(gm,dims,cdim),old.GENERATOR_CKPT.read_bytes())
 cm=Critic();cp=serialization.from_bytes(merge_initialized(cm,dims,cdim,critic=True),old.CRITIC_CKPT.read_bytes());return gm,gp,cm,cp

def freeze():
 rt=runtimes();norm=old.load(old.NORMALIZATION);gm,gp,cm,cp=load_models(norm);rows=[]
 state_manifest={'created_before_proposals':True,'created_before_outcomes':True,'counts':{},'states':[]}
 for sc,r in rt.items():
  state_manifest['counts'][sc]=len(r.states)
  for s in r.states:state_manifest['states'].append({'scenario':sc,'state_uid':s['uid'],'state_id':s['alias'],'physical':s['physical'],'metadata':s['metadata']})
 dump('fresh_test_manifests.json',state_manifest)
 for sc,r in rt.items():
  n=norm['scenarios'][sc]
  for s in r.states:
   flat,env=r.conditioning(s);hh=(np.asarray(flat,np.float32)-np.asarray(n['h_mean'],np.float32))/np.asarray(n['h_std'],np.float32);cc=context_vector(env,norm,sc)
   raw=np.asarray(gm.apply(gp,jnp.asarray(hh[None]),jnp.asarray(cc[None]),method=getattr(gm,METHOD[sc])))[0];mean=np.asarray(eta_mean(jnp.asarray(raw)),float)
   rng=np.random.default_rng(stable_int('generator-v1-proposals',41,sc,s['uid']));samples=np.asarray(eta_from_noise(jnp.asarray(raw)[None].repeat(K,0),jnp.asarray(rng.standard_normal((K,3)),jnp.float32)),float);etas=np.asarray([mean,*samples])
   scores=1/(1+np.exp(-np.asarray(cm.apply(cp,jnp.asarray(np.repeat(hh[None],5,0)),jnp.asarray(np.repeat(cc[None],5,0)),jnp.asarray((etas-CENTER)/RADIUS),method=getattr(cm,METHOD[sc])))))
   rows.append({'scenario':sc,'state_uid':s['uid'],'state_id':s['alias'],'mean':mean.tolist(),'samples':samples.tolist(),'critic_scores':scores.tolist(),'critic_index':int(np.argmax(scores)),'proposal_seed':stable_int('generator-v1-proposals',41,sc,s['uid'])})
 manifest={'created_before_outcomes':True,'K_FINAL':4,'eta_zero_adopted':False,'states':rows,'generator_hash':filehash(old.GENERATOR_CKPT),'critic_hash':filehash(old.CRITIC_CKPT),'safety_hashes':{sc:r.safety_hash for sc,r in rt.items()},'fair_eta':FAIR_ETA}
 dump('fresh_proposals.json',manifest);dump('frozen_hashes.json',{'proposal_manifest':filehash(OUT/'fresh_proposals.json'),'state_manifest':filehash(OUT/'fresh_test_manifests.json'),'generator':manifest['generator_hash'],'critic':manifest['critic_hash'],'safety':manifest['safety_hashes']});return manifest

def lookup(runtime,state,eta,chain,seed):return old.lookup(runtime,state,eta,chain,[seed])
def tasks(rt,manifest):
 by={(x['scenario'],x['state_uid']):x for x in manifest['states']};out=[]
 for sc,r in rt.items():
  for s in r.states:
   p=by[(sc,s['uid'])];vals=[('B0',[0.,0.,0.],'hard_safety_q16'),('B1_FAIR',FAIR_ETA[sc],'orthoflow3'),('mean',p['mean'],'orthoflow3')]+[(f'sample_{i}',e,'orthoflow3') for i,e in enumerate(p['samples'])]
   seen=set()
   for name,eta,chain in vals:
    k=eta_identity(eta)[0]
    if k in seen:continue
    seen.add(k)
    for seed in SEEDS:out.append((r,s,name,eta,chain,seed))
 return out

def preflight():
 m=json.loads((OUT/'fresh_proposals.json').read_text());rt=runtimes();total=Counter();detail=[]
 for r,s,name,eta,chain,seed in tasks(rt,m):
  rows,missing=lookup(r,s,eta,chain,seed);total['requested']+=1;total['reused']+=bool(rows);total['missing']+=bool(missing)
  if missing:detail.append({'scenario':r.name,'state_uid':s['uid'],'candidate':name,'seed':seed})
 out={'database':str(old.DB_PATH),'summary':dict(total),'missing':detail};dump('cache_preflight.json',out);return out

def run(shard,shards):
 m=json.loads((OUT/'fresh_proposals.json').read_text());rt=runtimes();alltasks=tasks(rt,m);selected=[x for i,x in enumerate(alltasks) if i%shards==shard];physical=reused=0;raw=OUT/'raw'/f'shard{shard}of{shards}.jsonl'
 for i,(r,s,name,eta,chain,seed) in enumerate(selected):
  rows,missing=lookup(r,s,eta,chain,seed)
  if not missing:reused+=1;continue
  for attempt in range(4):
   z=r.rollout(s,np.asarray(eta,float),seed,chain);z.update({'candidate':name,'attempt':attempt,'fresh_test':True});old.insert_rollout(r,z,raw);physical+=1
   if not z['numerical_failure']:break
  if (i+1)%100==0:print(json.dumps({'shard':shard,'done':i+1,'physical':physical,'reused':reused}),flush=True)
 out={'shard':shard,'tasks':len(selected),'physical':physical,'reused':reused};dump(f'run_shard{shard}of{shards}.json',out);return out

def collect(r,s,eta,chain):
 rows=[]
 for seed in SEEDS:
  found,missing=lookup(r,s,eta,chain,seed)
  if found:row=dict(found[seed])
  else:
   eu=eta_identity(eta)[0];sk=canonical({'future_index':seed})
   with connect(True) as con:x=con.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?',(s['uid'],eu,r.controllers[chain]['uid'],sk)).fetchone()
   if x is None:raise RuntimeError((r.name,s['uid'],chain,seed))
   row=dict(x)
  rows.append(row)
 return rows

def robust(rows):
 succ=sum(x['success'] for x in rows);valid=[x for x in rows if not x['numerical_failure']];fail=len(valid)-succ
 return True if succ>=15 else False if fail>=2 else None

def finalize():
 m=json.loads((OUT/'fresh_proposals.json').read_text());rt=runtimes();by={(x['scenario'],x['state_uid']):x for x in m['states']};per=[]
 for sc,r in rt.items():
  for s in r.states:
   p=by[(sc,s['uid'])];b0=collect(r,s,[0,0,0],'hard_safety_q16');b1=collect(r,s,FAIR_ETA[sc],'orthoflow3');cand=[('mean',p['mean']),*[(f'sample_{i}',e) for i,e in enumerate(p['samples'])]];ce={n:collect(r,s,e,'orthoflow3') for n,e in cand}
   score={n:sum(x['success'] for x in rr) for n,rr in ce.items()};oracle=max(cand,key=lambda z:(score[z[0]],-cand.index(z)))[0];critic=cand[p['critic_index']][0]
   methods={'B0':b0,'B1_FAIR':b1,'B2':ce['mean'],'B4':ce[oracle],'B5':ce[critic]};base=robust(b0);row={'scenario':sc,'state_uid':s['uid'],'oracle_candidate':oracle,'critic_candidate':critic,'critic_scores':p['critic_scores']}
   for name,rr in methods.items():
    row[name]={'successes':sum(x['success'] for x in rr),'Q16_lower':sum(x['success'] for x in rr)/16,'robust':robust(rr),'numerical':sum(x['numerical_failure'] for x in rr),'collisions':sum(x['collision'] for x in rr),'timeouts':sum(x['timeout'] for x in rr),'rescue':base is False and robust(rr) is True,'break':base is True and robust(rr) is False}
   per.append(row)
 summary={}
 for sc in rt:
  x=[r for r in per if r['scenario']==sc];summary[sc]={}
  for method in ('B0','B1_FAIR','B2','B4','B5'):
   summary[sc][method]={'states':len(x),'robust':sum(r[method]['robust'] is True for r in x),'nonrobust':sum(r[method]['robust'] is False for r in x),'unresolved':sum(r[method]['robust'] is None for r in x),'mean_Q16_lower':float(np.mean([r[method]['Q16_lower'] for r in x])),'rescue':sum(r[method]['rescue'] for r in x),'break':sum(r[method]['break'] for r in x),'collision_seeds':sum(r[method]['collisions'] for r in x),'numerical_seeds':sum(r[method]['numerical'] for r in x),'timeout_seeds':sum(r[method]['timeouts'] for r in x)}
  summary[sc]['attribution']={'same_candidate':sum(r['oracle_candidate']==r['critic_candidate'] for r in x),'oracle_robust_critic_nonrobust':sum(r['B4']['robust'] is True and r['B5']['robust'] is False for r in x),'mean_Q_gap':float(np.mean([r['B4']['Q16_lower']-r['B5']['Q16_lower'] for r in x]))}
 dump('fresh_test_per_state.json',per);dump('fresh_test_results.json',summary)
 with connect(True) as con:audit={'integrity_check':con.execute('PRAGMA integrity_check').fetchone()[0],'foreign_key_violations':[list(x) for x in con.execute('PRAGMA foreign_key_check')],'rollouts':con.execute('SELECT COUNT(*) FROM rollout').fetchone()[0]}
 dump('database_integrity.json',audit);dump('cache_postflight.json',preflight());return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','preflight','run','finalize'));p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
 out=freeze() if a.stage=='freeze' else preflight() if a.stage=='preflight' else run(a.shard,a.shards) if a.stage=='run' else finalize();print(json.dumps(out if a.stage!='freeze' else {'states':len(out['states'])},indent=2))
if __name__=='__main__':main()
