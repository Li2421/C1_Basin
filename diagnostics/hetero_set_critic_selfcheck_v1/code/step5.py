"""Final scoring: validation (ensemble) and untouched test set. All choices read from FINAL_FROZEN.json."""
import json,numpy as np
from tab import load,KEYS
Z=json.load(open('FINAL_FROZEN.json'));CH=Z['choices'];F=list(range(8,24))
T=load(['design.jsonl','extra.jsonl','fresh.jsonl','test_menu.jsonl_conv'] if False else ['design.jsonl','extra.jsonl','fresh.jsonl'])
def readk(f,K=16,n=24):
 out={}
 for l in open(f):
  d=json.loads(l);t=out.setdefault(d['scene'],np.full((3,n,K if d['key'][0]=='p' else 49),np.nan))
  t[d['controller'],d['state'],int(d['key'][1:]) if d['key'][0]=='p' else KEYS.index(d['key'][5:])]=d['success']
 return out
VP=readk('ens_val.jsonl');TM=readk('test_menu.jsonl');TP=readk('test_props.jsonl');G8=readk('test_gate.jsonl',n=24)
for n,t in TM.items():t[:,:8]=G8[n][:,:8]
CR_=json.load(open('CRITIC_VALIDATION.json'))
def table(names,menu,props,label):
 rows={}
 def m(v,k):return menu[v][:,F,KEYS.index(k)].mean(0)
 for v in names:
  c=CH[v];r=dict(global_=m(v,c['global_']),knn1=m(v,c['knn1']),knn3=m(v,c['knn3']))
  r['menu_critic']=m(v,c['menu_critic'] if c['menu_critic'] else CR_['critic_list']['choices'][v])
  r['gen+critic16']=props[v][:,F,c['ens16']].mean(0);r['gen+critic8']=props[v][:,F,c['ens8']].mean(0)
  r['union']=props[v][:,F,c['union'][1]].mean(0) if c['union'][0]=='p' else m(v,c['union'][1])
  A=list(range(8,16));B=list(range(16,24));r['honest_best16_ref']=np.array([np.nan]*16);hb=np.mean([props[v][:,Zs,int(props[v][:,X].mean((0,1)).argmax())].mean() for X,Zs in [(A,B),(B,A)]])
  r['honest_best16_ref'][:]=hb
  # own best (chosen on states 0-7) for heterogeneity check
  o=int(menu[v][:,:8].mean((0,1)).argmax());r['own_best_from_states0_7']=menu[v][:,F,o].mean(0)
  rows[v]=r
 meth=list(rows[names[0]]);M=len(names);arr={k:np.array([rows[v][k] for v in names]) for k in meth}
 rng=np.random.default_rng(2026102797);si=rng.integers(16,size=(5000,M,16));sc=rng.integers(M,size=(5000,M));sh=rng.integers(16,size=(5000,M,16))
 def ci(d):return np.percentile(d[np.arange(M)[None,:,None],si].mean((1,2)),[2.5,97.5]),np.percentile(d[sc[:,:,None],sh].mean((1,2)),[2.5,97.5])
 print(f'\n===== {label}: {M} scenes, global eta {CH[names[0]]["global_"]}')
 res={}
 for k,a in arr.items():
  d=a-arr['global_'];c1,c2=ci(d);dk=a-arr['knn3'];k1,k2=ci(dk);res[k]=dict(rate=float(a.mean()),minus_global=float(d.mean()),ci_state=c1.tolist(),ci_scene_state=c2.tolist(),minus_knn3=float(dk.mean()),knn3_ci_state=k1.tolist(),knn3_ci_scene_state=k2.tolist())
  print(f"{k:24s} {a.mean()*100:6.2f} | -global {d.mean()*100:+6.2f} state [{c1[0]*100:+.2f},{c1[1]*100:+.2f}] scene [{c2[0]*100:+.2f},{c2[1]*100:+.2f}] | -knn3 {dk.mean()*100:+6.2f} state [{k1[0]*100:+.2f},{k1[1]*100:+.2f}]")
 for v in names:print(f"  {v:28s}"+' '.join(f"{k[:9]}={rows[v][k].mean()*100:5.1f}" for k in ['global_','knn3','menu_critic','gen+critic16','union','own_best_from_states0_7']))
 return res
out=dict(validation=table(Z['val'],T,VP,'VALIDATION (ensemble generator, retrained critic)'),test=table(Z['test'],TM,TP,'TEST (untouched siblings)'))
json.dump(out,open('FINAL_RESULTS.json','w'),indent=1)
