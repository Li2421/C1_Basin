from tab import *
import json
sp=json.load(open('SPLIT.json'));VA=sp['val'];G=json.load(open('GEN_FROZEN.json'));CV=json.load(open('CRITIC_VALIDATION.json'))
T=load(['design.jsonl','extra.jsonl','fresh.jsonl']);F=list(range(8,24))
Y={v:np.full((3,24,16),np.nan) for v in VA}
for l in open('gen.jsonl'):
 d=json.loads(l);Y[d['scene']][d['controller'],d['state'],d['proposal']]=d['success']
arr={}
def menu(m):return np.array([T[v][:,F,KEYS.index(CV[m]['choices'][v])].mean(0) for v in VA])
for m in ['global','knn3','critic_list']:arr[m]=menu(m)
arr['gen+critic_16']=np.array([Y[v][:,F,G[v]['pick16']].mean(0) for v in VA])
arr['gen+critic_8']=np.array([Y[v][:,F,G[v]['pick8']].mean(0) for v in VA])
arr['gen_only_mean0']=np.array([Y[v][:,F,0].mean(0) for v in VA])
arr['gen_proposal_avg']=np.array([Y[v][:,F,:].mean((0,2)) for v in VA])
arr['best_of_16_ref']=np.array([Y[v][:,F,int(Y[v][:,F].mean((0,1)).argmax())].mean(0) for v in VA])
assert all(np.isfinite(a).all() for a in arr.values())
rng=np.random.default_rng(2026102695);M,Sn=8,16
si=rng.integers(Sn,size=(5000,M,Sn));sc=rng.integers(M,size=(5000,M));sh=rng.integers(Sn,size=(5000,M,Sn))
def ci(d):return np.percentile(d[np.arange(M)[None,:,None],si].mean((1,2)),[2.5,97.5]),np.percentile(d[sc[:,:,None],sh].mean((1,2)),[2.5,97.5])
res={}
for m,a in arr.items():
 out=dict(rate=float(a.mean()))
 for ref in ['global','knn3','critic_list']:
  d=a-arr[ref];c1,c2=ci(d);out['minus_'+ref]=dict(diff=float(d.mean()),ci_state=c1.tolist(),ci_scene_state=c2.tolist())
 res[m]=out
 print(f"{m:18s} {a.mean()*100:6.2f} | -global {out['minus_global']['diff']*100:+6.2f} [{out['minus_global']['ci_state'][0]*100:+.2f},{out['minus_global']['ci_state'][1]*100:+.2f}] scene [{out['minus_global']['ci_scene_state'][0]*100:+.2f},{out['minus_global']['ci_scene_state'][1]*100:+.2f}] | -knn3 {out['minus_knn3']['diff']*100:+6.2f} [{out['minus_knn3']['ci_state'][0]*100:+.2f},{out['minus_knn3']['ci_state'][1]*100:+.2f}] | -menu critic {out['minus_critic_list']['diff']*100:+6.2f} [{out['minus_critic_list']['ci_state'][0]*100:+.2f},{out['minus_critic_list']['ci_state'][1]*100:+.2f}]")
print()
for i,v in enumerate(VA):
 g=G[v];e16=np.round(g['proposals'][g['pick16']],2).tolist();b=int(Y[v][:,F].mean((0,1)).argmax())
 print(f"{v:26s} global {arr['global'][i].mean()*100:5.1f} knn3 {arr['knn3'][i].mean()*100:5.1f} menuCritic {arr['critic_list'][i].mean()*100:5.1f} | gen+critic16 {arr['gen+critic_16'][i].mean()*100:5.1f} {e16} | gen+critic8 {arr['gen+critic_8'][i].mean()*100:5.1f} | best16 {arr['best_of_16_ref'][i].mean()*100:5.1f} {np.round(g['proposals'][b],2).tolist()}")
json.dump(res,open('GEN_VALIDATION.json','w'),indent=1)
