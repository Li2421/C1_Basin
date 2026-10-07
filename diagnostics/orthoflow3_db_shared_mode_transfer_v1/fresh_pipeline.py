#!/usr/bin/env python3
from __future__ import annotations
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
H=Path(__file__).parent;POOL=H.parent/'double_bottleneck_initial_state_coverage/data/train_pool'
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
 fields=fields or list(rows[0]);
 with open(H/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
class MLP(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(64)(x));x=nn.silu(nn.Dense(64)(x));return nn.Dense(12)(x)
def plans(name,tasks):
 groups=defaultdict(list)
 for t in tasks:groups[(t['controller'],t['mode_id'])].append(t)
 load=[0]*6;owner={}
 for g,z in sorted(groups.items(),key=lambda q:(-len(q[1]),q[0])):j=int(np.argmin(load));owner[g]=j;load[j]+=len(z)
 p=H/'plans'/name;p.mkdir(parents=True,exist_ok=True)
 for j in range(6):
  with open(p/f'shard{j}.jsonl','w') as f:
   for t in tasks:
    if owner[(t['controller'],t['mode_id'])]==j:f.write(json.dumps(t)+'\n')
 return load
def manifest():
 main=json.load(open(H/'db_state_split.json'));used={s['source_group'] for s in main['states']};m=json.load(open(POOL/'manifest.json'));fam=sorted({r['family_id'] for r in m['files'] if r['family_id'] not in used},key=lambda s:hashlib.sha256(('db_mode_transfer_fresh|'+s).encode()).hexdigest())[:32]
 states=[{'state_id':f'DB_FRESH_{i:03d}','split':'fresh','family_id':f,'source_group':f,'dataset':str(POOL),'initial_flow_root':2026092906,'future_root':2026092907,'rng_namespace':40000+i,'selection':'outcome-blind SHA256 from unused train-pool source groups','h_schema':'obs72+current_raw_flow8'} for i,f in enumerate(fam)]
 (H/'fresh_state_manifest.json').write_text(json.dumps({'frozen_before_outcomes':True,'states':states},indent=2,sort_keys=True)+'\n');tasks=[]
 for s in states:tasks.append({'state_id':s['state_id'],'eta':[0.,0.,0.],'future_index':0,'phase':'fresh_feature','probe_id':s['state_id']+'_feature','controller':'fresh_safety','mode_id':-1,'split':'fresh'})
 print(json.dumps({'states':len(states),'feature_loads':plans('fresh_feature',tasks)},indent=2))
def predict():
 fs=json.load(open(H/'fresh_state_manifest.json'))['states'];a=np.load(H/'state_features.npz');mean=a['mean'];std=a['std'];X=[]
 for s in fs:X.append((np.asarray(json.load(open(H/'raw'/f"conditioning_{s['state_id']}.json"))['h_feature'])-mean)/std)
 X=np.asarray(X,np.float32);sel=json.load(open(H/'selected_selector.json'));cal=json.load(open(H/'calibration.json'));thr=json.load(open(H/'selected_threshold.json'));model=MLP();tmp=model.init(jax.random.PRNGKey(0),jnp.zeros((1,80)));params=serialization.from_bytes(tmp,Path(sel['checkpoint']).read_bytes());log=np.asarray(model.apply(params,jnp.asarray(X)));T=cal['temperature'] if cal['used'] else 1.;P=1/(1+np.exp(-np.clip(log/T,-30,30)));summary=json.load(open(H/'test_summary.json'));fixed=int(summary['train_prior_best_fixed_mode']);E=np.array(json.load(open(H/'selected_transform.json'))['transform']['eta']);rows=[];tasks=[]
 for i,s in enumerate(fs):
  m=int(np.argmax(P[i]));conf=float(P[i,m]);ab=conf<float(thr['tau']);rows.append({'state_id':s['state_id'],'fixed_mode':fixed,'selected_mode':m,'confidence':conf,'abstained':ab})
  needed={fixed}|(set() if ab else {m})
  for mm in sorted(needed):
   for fi in range(64):tasks.append({'state_id':s['state_id'],'eta':E[mm].tolist(),'future_index':fi,'phase':'fresh_eval','probe_id':f'{s["state_id"]}_mode{mm}_{fi}','controller':'fresh_mode','mode_id':mm,'split':'fresh'})
  for fi in range(64):tasks.append({'state_id':s['state_id'],'eta':[0.,0.,0.],'future_index':fi,'phase':'fresh_eval','probe_id':f'{s["state_id"]}_safety_{fi}','controller':'fresh_safety','mode_id':-1,'split':'fresh'})
 write('fresh_predictions.csv',rows);print(json.dumps({'tasks':len(tasks),'loads':plans('fresh_eval',tasks),'fixed_mode':fixed,'abstentions':sum(r['abstained'] for r in rows)},indent=2))
def agg(v):
 o=Counter(str(r['outcome']) for r in v);s=[r for r in v if r['success']];return {'successes':len(s),'trials':len(v),'Q64':len(s)/len(v),'B63':len(s)>=63,'deadlock':sum(o[q] for q in ('strict_deadlock','safe_deadlock','deadlock')),'timeout':o['timeout'],'collision':sum('collision' in str(r['outcome']) for r in v),'J_def':float(np.mean([r['J_def'] for r in v])) if v and 'J_def' in v[0] else '','episode_length':float(np.mean([r['episode_steps'] for r in v]))}
def aggregate():
 pred={r['state_id']:r for r in read(H/'fresh_predictions.csv')};z=[]
 for p in (H/'raw').glob('fresh_*.jsonl'):z += [json.loads(x) for x in open(p) if x.strip()]
 by=defaultdict(dict)
 for r in z:by[(r['state_id'],r['controller'],int(r['mode_id']))][int(r['future_index'])]=r
 rows=[]
 for sid,p in pred.items():
  safe=agg([by[(sid,'fresh_safety',-1)][i] for i in range(64)]);fixed=agg([by[(sid,'fresh_mode',int(p['fixed_mode']))][i] for i in range(64)]);sel=safe if p['abstained']=='True' else agg([by[(sid,'fresh_mode',int(p['selected_mode']))][i] for i in range(64)]);rows.append({'state_id':sid,**{f'safety_{k}':v for k,v in safe.items()},**{f'fixed_{k}':v for k,v in fixed.items()},**{f'selector_{k}':v for k,v in sel.items()},'fixed_mode':p['fixed_mode'],'selected_mode':p['selected_mode'],'abstained':p['abstained']})
 write('fresh_replication.csv',rows)
 def summary(c):return {'controller':c,'B63_states':sum(str(r[f'{c}_B63']).lower()=='true' for r in rows),'states':len(rows),'mean_Q64':float(np.mean([float(r[f'{c}_Q64']) for r in rows])),'success':sum(int(r[f'{c}_successes']) for r in rows),'trials':sum(int(r[f'{c}_trials']) for r in rows),'deadlock':sum(int(r[f'{c}_deadlock']) for r in rows),'timeout':sum(int(r[f'{c}_timeout']) for r in rows),'collision':sum(int(r[f'{c}_collision']) for r in rows),'J_def':float(np.mean([float(r[f'{c}_J_def']) for r in rows])),'episode_length':float(np.mean([float(r[f'{c}_episode_length']) for r in rows]))}
 pairs={'selector_vs_fixed':{'rescue':0,'break':0},'selector_vs_safety':{'rescue':0,'break':0},'fixed_vs_safety':{'rescue':0,'break':0}}
 for sid,p in pred.items():
  ctl='fresh_safety' if p['abstained']=='True' else 'fresh_mode';mm=-1 if p['abstained']=='True' else int(p['selected_mode']);fm=int(p['fixed_mode'])
  for fi in range(64):
   a=bool(by[(sid,'fresh_safety',-1)][fi]['success']);f=bool(by[(sid,'fresh_mode',fm)][fi]['success']);s=bool(by[(sid,ctl,mm)][fi]['success'])
   pairs['selector_vs_fixed']['rescue']+=(not f) and s;pairs['selector_vs_fixed']['break']+=f and not s
   pairs['selector_vs_safety']['rescue']+=(not a) and s;pairs['selector_vs_safety']['break']+=a and not s
   pairs['fixed_vs_safety']['rescue']+=(not a) and f;pairs['fixed_vs_safety']['break']+=a and not f
 counts=Counter('safety' if p['abstained']=='True' else str(p['selected_mode']) for p in pred.values());v=np.array(list(counts.values()),float);pp=v/v.sum();entropy=float(-np.sum(pp*np.log(pp))/np.log(12)) if len(v)>1 else 0.
 out={'controllers':[summary(c) for c in ('safety','fixed','selector')],'paired':pairs,'selector_mode_counts':dict(counts),'normalized_selection_entropy':entropy};(H/'fresh_summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2))
def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['manifest','predict','aggregate']);q=a.parse_args();{'manifest':manifest,'predict':predict,'aggregate':aggregate}[q.stage]()
if __name__=='__main__':main()
