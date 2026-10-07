#!/usr/bin/env python3
"""Freeze predictions/plans and aggregate CENTER-vs-MARGIN evaluations."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_basin_margin_learning_v1'
DIRECT=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1';PREV=ROOT/'diagnostics/orthoflow3_large_margin_ball_transfer_v1';PILOT=ROOT/'diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1'
AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75]);GAMMA=.8;SEEDS=(17,23,41);SHARDS=6
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def read_csv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rr,fields=None):
 if fields is None:fields=list(rr[0])
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)
def read_jsonl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
class G(nn.Module):
 @nn.compact
 def __call__(self,x):
  x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return nn.Dense(3)(x)
class GLowJ(nn.Module):
 @nn.compact
 def __call__(self,x):
  low=jnp.array([0.,-.5,0.]);high=jnp.array([1.25,.5,.75]);x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return low+nn.sigmoid(nn.Dense(3)(x))*(high-low)
def halfspaces():
 return np.array([[float(x[k]) for k in ('n1','n2','n3','b')] for x in read_csv(HERE/'eligible_new_anchor_source/ebridge_halfspaces.csv')])
def inside(x):
 eq=halfspaces();return bool(np.all(eq[:,:3]@x+eq[:,3]<=1e-9))
def load_model(path):
 m=G();t=m.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));return m,serialization.from_bytes(t,Path(path).read_bytes())
def build_state_manifest():
 data=read_csv(HERE/'verified_ball_dataset.csv');by={}
 for x in json.load(open(PILOT/'frozen_state_manifest.json'))['states']:by[x['state_id']]=x
 for d in (HERE/'anchor_builds').iterdir():
  x=json.load(open(d/'frozen_state_manifest.json'))['states'][0];by[x['state_id']]=x
 for x in json.load(open(PREV/'neighbor_manifest.json'))['neighbors']:by[x['neighbor_id']]=x
 for x in json.load(open(HERE/'new_anchor_transfers/neighbor_manifest.json'))['neighbors']:by[x['neighbor_id']]=x
 out=[]
 for i,r in enumerate(data):
  x=dict(by[r['state_id']]);x['state_id']=r['state_id'];x['dataset_index']=i;x['r_ball']=float(r['r_ball']);x['c']=[float(r['c1']),float(r['c2']),float(r['c3'])];x['anchor_family']=r['anchor_family'];x['label_kind']=r['label_kind'];out.append(x)
 dump(HERE/'evaluation_state_manifest.json',{'future_root_seed':2026092801,'states':out,'frozen_dataset_sha256':sha(HERE/'verified_ball_dataset.csv')})
def arrays():
 a=np.load(HERE/'learning_arrays.npz');return a,read_csv(HERE/'verified_ball_dataset.csv')
def predict_new(arm,seed,indices):
 s=json.load(open(HERE/f'g_{arm}/runs/{arm}_seed{seed}_summary.json'));m,p=load_model(s['checkpoint']);a,_=arrays();yn=np.asarray(m.apply(p,jnp.asarray(a['x'][indices])));return yn,AFF+SCALE*yn,s
def predict_lowj(indices):
 sel=json.load(open(DIRECT/'selected_checkpoint.json'));norm=json.load(open(DIRECT/'normalization.json'));features=np.load(HERE/'learning_features.npz')['features'];x=((features[indices]-np.asarray(norm['h_mean']))/np.asarray(norm['h_std'])).astype(np.float32)
 m=GLowJ();t=m.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));p=serialization.from_bytes(t,Path(sel['checkpoint']).read_bytes());return np.asarray(m.apply(p,jnp.asarray(x))),sel
def write_plan(name,tasks):
 d=HERE/'plans'/name;d.mkdir(parents=True,exist_ok=True)
 # Stable round-robin by state and controller to balance, while seed identities remain matched.
 groups=sorted({(x['state_id'],x['controller']) for x in tasks});owner={g:i%SHARDS for i,g in enumerate(groups)}
 for s in range(SHARDS):
  with open(d/f'shard{s}.jsonl','w') as f:
   for x in tasks:
    if owner[(x['state_id'],x['controller'])]==s:f.write(json.dumps(x,sort_keys=True)+'\n')
 dump(d/'summary.json',{'name':name,'tasks':len(tasks),'shards':SHARDS,'future_root_seed':2026092801,'files':[{'shard':s,'sha256':sha(d/f'shard{s}.jsonl')} for s in range(SHARDS)]})
def make_val():
 build_state_manifest();a,data=arrays();idx=np.where(a['splits']=='val')[0];out=[];tasks=[]
 for arm in ('center','margin'):
  for seed in SEEDS:
   yn,yp,s=predict_new(arm,seed,idx)
   for ii,ytn,eta in zip(idx,yn,yp):
    r=data[ii];c=np.array([float(r[k]) for k in ('c1','c2','c3')]);cn=(c-AFF)/SCALE;rho=float(np.linalg.norm(ytn-cn)/float(r['r_ball']));ctl=f'{arm}_seed{seed}'
    out.append({'state_id':r['state_id'],'arm':arm,'seed':seed,'controller':ctl,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'eta_norm1':ytn[0],'eta_norm2':ytn[1],'eta_norm3':ytn[2],'rho':rho,'inside_ball':rho<=1,'inside_retained':rho<=GAMMA,'inside_Ebridge':inside(ytn),'checkpoint_sha256':s['checkpoint_sha256']})
    for fi in range(16):tasks.append({'state_id':r['state_id'],'controller':ctl,'arm':arm,'seed':seed,'eta':eta.tolist(),'future_index':fi,'phase':'val16','checkpoint_sha256':s['checkpoint_sha256']})
 write_csv(HERE/'val_predictions.csv',out);write_plan('val16',tasks);print(json.dumps({'val_predictions':len(out),'tasks':len(tasks)}))
def all_rollouts(name):
 rr=[]
 for p in (HERE/'raw'/name).glob('shard*.jsonl'):rr+=read_jsonl(p)
 return rr
def aggregate_val():
 pred=read_csv(HERE/'val_predictions.csv');roll=all_rollouts('val16');rowsout=[];selections={}
 for arm in ('center','margin'):
  candidates=[]
  for seed in SEEDS:
   ctl=f'{arm}_seed{seed}';rs=[x for x in roll if x['controller']==ctl];pp=[x for x in pred if x['controller']==ctl];by=defaultdict(list)
   for x in rs:by[x['state_id']].append(x)
   state16=sum(sum(y['success'] for y in v)==16 for v in by.values());success=sum(x['success'] for x in rs);dead=sum(x['outcome']=='safe_deadlock' for x in rs);tout=sum(x['outcome']=='timeout' for x in rs);coll=sum(x['outcome']=='collision' for x in rs);incl=sum(x['inside_ball']=='True' for x in pp);retain=sum(x['inside_retained']=='True' for x in pp);j=[x['J_def'] for x in rs if x['success']]
   s=json.load(open(HERE/f'g_{arm}/runs/{arm}_seed{seed}_summary.json'))
   rec={'arm':arm,'seed':seed,'total_successes':success,'trials':len(rs),'states_16of16':state16,'inside_ball_states':incl,'inside_retained_states':retain,'deadlock':dead,'timeout':tout,'collision':coll,'successful_J_def_mean':float(np.mean(j)) if j else None,'successful_J_def_median':float(np.median(j)) if j else None,'family_val_loss':s['best_val_loss'],'checkpoint':s['checkpoint'],'checkpoint_sha256':s['checkpoint_sha256'],'best_epoch':s['best_epoch']};candidates.append(rec);rowsout.append(rec)
  candidates.sort(key=lambda x:(-x['total_successes'],-x['states_16of16'],-x['inside_ball_states'],x['deadlock'],-x['inside_retained_states'],x['family_val_loss'],x['seed']))
  win=candidates[0];selections[arm]=win;d=HERE/f'g_{arm}';write_csv(d/'val_closedloop.csv',[x for x in rowsout if x['arm']==arm]);dump(d/'selected_checkpoint.json',win);(d/'checkpoint_sha256.txt').write_text(win['checkpoint_sha256']+'\n')
 dump(HERE/'val_selection.json',{'lexicographic_rule':['total_successes','states_16of16','inside_ball_states','fewer_deadlocks','inside_retained_states','family_val_loss','lower_seed'],'selected':selections,'frozen_before_test':True});print(json.dumps(selections,indent=2))
def make_test():
 a,data=arrays();idx=np.where(a['splits']=='test')[0];sel=json.load(open(HERE/'val_selection.json'))['selected'];out=[];tasks=[]
 for arm in ('center','margin'):
  seed=int(sel[arm]['seed']);yn,yp,s=predict_new(arm,seed,idx)
  for ii,ytn,eta in zip(idx,yn,yp):
   r=data[ii];cn=(np.array([float(r[k]) for k in ('c1','c2','c3')])-AFF)/SCALE;dist=float(np.linalg.norm(ytn-cn));rho=dist/float(r['r_ball']);ctl=f'g_{arm}'
   out.append({'state_id':r['state_id'],'model':ctl,'seed':seed,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'normalized_center_distance':dist,'rho':rho,'signed_margin':1-rho,'inside_ball':rho<=1,'inside_retained':rho<=GAMMA,'inside_Ebridge':inside(ytn),'r_ball':r['r_ball'],'checkpoint_sha256':s['checkpoint_sha256']})
   for fi in range(64):tasks.append({'state_id':r['state_id'],'controller':ctl,'arm':arm,'seed':seed,'eta':eta.tolist(),'future_index':fi,'phase':'test64','checkpoint_sha256':s['checkpoint_sha256']})
 # Frozen low-J is feature-compatible: same exact 214-D deployment builder and state-conditioning features.
 low,ls=predict_lowj(idx)
 for ii,eta in zip(idx,low):
  r=data[ii];out.append({'state_id':r['state_id'],'model':'g_lowj','seed':ls['seed'],'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'normalized_center_distance':'','rho':'','signed_margin':'','inside_ball':'','inside_retained':'','inside_Ebridge':inside((eta-AFF)/SCALE),'r_ball':r['r_ball'],'checkpoint_sha256':ls['checkpoint_sha256']})
  for fi in range(64):tasks.append({'state_id':r['state_id'],'controller':'g_lowj','arm':'lowj','seed':ls['seed'],'eta':eta.tolist(),'future_index':fi,'phase':'test64','checkpoint_sha256':ls['checkpoint_sha256']})
 write_csv(HERE/'test_geometric_predictions.csv',out);write_plan('test64',tasks);dump(HERE/'frozen_lowj_reference.json',{'status':'COMPATIBLE','reason':'same authoritative 214-D deployment feature and fixed-current-Flow conditioning','checkpoint':ls,'checkpoint_sha256_verified':sha(ls['checkpoint'])==ls['checkpoint_sha256'],'retrained':False});print(json.dumps({'geometry_rows':len(out),'tasks':len(tasks)}))
def make_perturb():
 geo=[x for x in read_csv(HERE/'test_geometric_predictions.csv') if x['model'] in ('g_center','g_margin')];tasks=[];pts=[];dirs=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1]],float);dirs/=np.linalg.norm(dirs,axis=1)[:,None]
 for x in geo:
  base=(np.array([float(x[k]) for k in ('eta1','eta2','eta3')])-AFF)/SCALE;r=float(x['r_ball'])
  for mag in (.25,.50):
   for di,d in enumerate(dirs):
    yn=base+mag*r*d;eta=AFF+SCALE*yn;ok=inside(yn);pid=f"{x['model']}__{x['state_id']}__m{mag:.2f}__d{di}"
    pts.append({'point_id':pid,'state_id':x['state_id'],'model':x['model'],'magnitude_r':mag,'direction_index':di,'inside_Ebridge':ok,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2]})
    if ok:
     for fi in range(8):tasks.append({'state_id':x['state_id'],'controller':x['model'],'arm':x['model'][2:],'seed':int(x['seed']),'eta':eta.tolist(),'future_index':fi,'phase':'perturb','point_id':pid,'magnitude_r':mag,'direction_index':di,'checkpoint_sha256':x['checkpoint_sha256']})
 write_csv(HERE/'perturbation_points.csv',pts);write_plan('perturb',tasks);print(json.dumps({'points':len(pts),'valid_points':sum(x['inside_Ebridge'] for x in pts),'tasks':len(tasks)}))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['make-val','aggregate-val','make-test','make-perturb']);a=ap.parse_args()
 {'make-val':make_val,'aggregate-val':aggregate_val,'make-test':make_test,'make-perturb':make_perturb}[a.stage]()
if __name__=='__main__':main()
