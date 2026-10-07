#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
import sys;sys.path.insert(0,str(Path(__file__).parent))
from train import Residual,bounded
H=Path(__file__).parent;D=H.parent;SH=D/'orthoflow3_shared_eta_codebook_v1';FRESH=D/'orthoflow3_fresh_selector_vs_fixed_v1';AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75])
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
 p=H/name;fields=fields or (list(rows[0]) if rows else ['state_id'])
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def prepare():
 ck=json.load(open(H/'selected_checkpoint.json'));model=Residual();tmp=model.init(jax.random.PRNGKey(0),jnp.zeros((1,214)),jnp.zeros((1,),jnp.int32));p=serialization.from_bytes(tmp,Path(ck['checkpoint']).read_bytes());features=np.load(FRESH/'fresh_state_features.npz')['features'];norm=json.load(open(SH/'normalization.json'));x=((features-np.array(norm['h_mean']))/np.array(norm['h_std'])).astype(np.float32);ms=read(FRESH/'mode_selection.csv');m=np.array([int(r['selected_mode']) for r in ms]);fa=np.load(H/'frozen_arrays.npz');anchors=(fa['anchors']-AFF)/SCALE;a=np.stack([anchors[q] for q in m]);raw=model.apply(p,jnp.asarray(x),jnp.asarray(m));d=np.asarray(bounded(raw,jnp.asarray(a),jnp.asarray(fa['halfspace_A']),jnp.asarray(fa['halfspace_b'])));pred=a+d;eta=AFF+SCALE*pred;viol=np.max(fa['halfspace_A']@pred.T+fa['halfspace_b'][:,None],axis=0)
 rows=[];tasks=[]
 for r,mi,di,zi,ei,vi in zip(ms,m,d,pred,eta,viol):
  rows.append(dict(state_id=r['state_id'],mode_id=mi,confidence=r['confidence'],anchor_eta1=fa['anchors'][mi,0],anchor_eta2=fa['anchors'][mi,1],anchor_eta3=fa['anchors'][mi,2],delta_z1=di[0],delta_z2=di[1],delta_z3=di[2],residual_norm=np.linalg.norm(di),eta1=ei[0],eta2=ei[1],eta3=ei[2],eta_norm=np.linalg.norm(ei),domain_max_violation=vi))
  for fi in range(64):tasks.append(dict(state_id=r['state_id'],controller='deformable',mode_id=int(mi),eta=ei.tolist(),future_index=fi,phase='deformable_test'))
 write('test_predictions.csv',rows)
 groups=defaultdict(list)
 for t in tasks:groups[t['state_id']].append(t)
 load=[0]*6;owner={}
 for sid in sorted(groups):j=int(np.argmin(load));owner[sid]=j;load[j]+=len(groups[sid])
 pp=H/'test_plans';pp.mkdir(exist_ok=True)
 for j in range(6):
  with (pp/f'shard{j}.jsonl').open('w') as f:
   for t in tasks:
    if owner[t['state_id']]==j:f.write(json.dumps(t,sort_keys=True)+'\n')
 dump('test_manifest.json',{'frozen_checkpoint':ck,'states':len(ms),'continuations':len(tasks),'shards':load,'max_domain_violation':float(np.max(viol)),'mode_identity_same_as_fixed_selector':True});print(json.dumps(json.load(open(H/'test_manifest.json')),indent=2))
def metric(v):
 out=Counter(x['outcome'] for x in v);k=sum(x['success'] for x in v);js=[x['J_def'] for x in v if x['success']];return dict(successes=k,trials=len(v),Q64=k/len(v),B63=k>=63,deadlock=out['safe_deadlock'],timeout=out['timeout'],collision=out['collision'],J_def=float(np.mean(js)) if js else '',episode_length=float(np.mean([x['continuation_steps'] for x in v])))
def aggregate():
 rr=[]
 for p in sorted((H/'test_raw').glob('shard*.jsonl')):rr += [json.loads(x) for x in open(p) if x.strip()]
 if len(rr)!=64*64:raise RuntimeError(('test incomplete',len(rr)))
 by=defaultdict(list)
 for r in rr:by[r['state_id']].append(r)
 fixed={r['state_id']:r for r in read(FRESH/'per_state_results.csv')};pred={r['state_id']:r for r in read(H/'test_predictions.csv')};fraw=[]
 for p in sorted((FRESH/'raw').glob('shard*.jsonl')):fraw += [json.loads(x) for x in open(p) if x.strip() and json.loads(x).get('controller')=='selector']
 fmap={(r['state_id'],int(r['future_index'])):r for r in fraw};rows=[];pairs=[]
 for sid,v in sorted(by.items()):
  q=metric(v);f=fixed[sid];base=dict(state_id=sid,mode_id=pred[sid]['mode_id'],residual_norm=pred[sid]['residual_norm'],moving_eta_norm=pred[sid]['eta_norm'])
  for k,x in q.items():base[f'deformable_{k}']=x
  for k in ('successes','trials','Q64','B63','deadlock','timeout','collision','J_def','episode_length'):base[f'fixed_{k}']=f[f'selector_{k}']
  rows.append(base);dm={int(x['future_index']):x for x in v};res=sum((not fmap[(sid,i)]['success']) and dm[i]['success'] for i in range(64));br=sum(fmap[(sid,i)]['success'] and (not dm[i]['success']) for i in range(64));pairs.append(dict(state_id=sid,mode_id=pred[sid]['mode_id'],residual_norm=pred[sid]['residual_norm'],rescue=res,break_count=br,net=res-br,fixed_Q64=f['selector_Q64'],deformable_Q64=q['Q64'],fixed_J_def=f['selector_J_def'],deformable_J_def=q['J_def']))
 write('test_results.csv',rows);write('fixed_vs_deformable.csv',pairs);dump('test_summary.json',{'states':len(rows),'fixed_B63':sum(r['fixed_B63']=='True' for r in rows),'deformable_B63':sum(r['deformable_B63'] for r in rows),'fixed_mean_Q64':float(np.mean([float(r['fixed_Q64']) for r in rows])),'deformable_mean_Q64':float(np.mean([float(r['deformable_Q64']) for r in rows])),'rescue':sum(r['rescue'] for r in pairs),'break':sum(r['break_count'] for r in pairs),'fixed_J_def':float(np.average([float(r['fixed_J_def']) for r in pairs],weights=[float(r['fixed_Q64']) for r in pairs])),'deformable_J_def':float(np.average([float(r['deformable_J_def']) for r in pairs if r['deformable_J_def']!=''],weights=[float(r['deformable_Q64']) for r in pairs if r['deformable_J_def']!=''])),'collision':sum(r['deformable_collision'] for r in rows)});print(json.dumps(json.load(open(H/'test_summary.json')),indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','aggregate']);a=ap.parse_args();prepare() if a.stage=='prepare' else aggregate()
if __name__=='__main__':main()
