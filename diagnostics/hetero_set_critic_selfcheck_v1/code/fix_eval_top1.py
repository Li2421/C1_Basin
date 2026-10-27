"""Train critic on menu + training-scene proposal labels; select among frozen validation proposals; score with existing labels."""
import json,pickle,numpy as np,jax
from tab import load,KEYS
import critic as CR,critic2 as C2
sp=json.load(open('SPLIT.json'));data=pickle.load(open('critic_data.pkl','rb'));TR=sp['train'];VA=sp['val']
G=json.load(open('GEN_TOP1_FROZEN.json'));PT=json.load(open('GEN_TOP1_LOSO_TRAIN_PROPOSALS.json'))
T=load(['design.jsonl','extra.jsonl','fresh.jsonl']);F=list(range(8,24))
tp={n:np.full((3,24,16),np.nan) for n in TR}
for l in open('gen_top1_train.jsonl'):d=json.loads(l);tp[d['scene']][d['controller'],d['state'],d['proposal']]=d['success']
Y={v:np.full((3,24,16),np.nan) for v in VA}
for l in open('gen_top1.jsonl'):d=json.loads(l);Y[d['scene']][d['controller'],d['state'],d['proposal']]=d['success']
nm=CR.norm_of(data,TR)
scenes={n:dict(E=np.concatenate([CR.MENU,np.array(PT[n])]),t=np.concatenate([data[n]['t'],tp[n]],-1),c=data[n]['c'],h=data[n]['h']) for n in TR}
print('train cells',sum(np.isfinite(s['t']).sum() for s in scenes.values()),'proposal cells',sum(np.isfinite(t).sum() for t in tp.values()))
L={v:dict(p=[],m=[]) for v in VA}
for seed in range(10):
 net,params=C2.train(scenes,nm,seed);ap=jax.jit(lambda p,x:net.apply(p,x))
 for v in VA:
  for key,E in [('p',np.array(G[v]['proposals'])),('m',CR.MENU)]:L[v][key].append(np.array([[np.asarray(ap(params,C2.Xe(data[v]['c'],data[v]['h'][s],k,nm,E))) for s in range(24)] for k in range(3)]))
def soft(f):p=np.exp(f-f.max(-1,keepdims=True));return (p/p.sum(-1,keepdims=True)).mean((0,1))
from scipy.stats import spearmanr
res={'gen+critic16_fixed':[],'gen+critic8_fixed':[],'union_fixed':[],'menu_fixed':[]};ch={}
for v in VA:
 fp=np.mean(L[v]['p'],0);fm=np.mean(L[v]['m'],0);yp=Y[v][:,F].mean((0,1));ym=T[v][:,F].mean((0,1))
 a16=int(soft(fp).argmax());a8=int(soft(fp[...,:8]).argmax());fu=np.concatenate([fp,fm],-1);au=int(soft(fu).argmax());am=int(soft(fm).argmax())
 res['gen+critic16_fixed'].append(Y[v][:,F,a16].mean(0));res['gen+critic8_fixed'].append(Y[v][:,F,a8].mean(0))
 res['union_fixed'].append(Y[v][:,F,au].mean(0) if au<16 else T[v][:,F,au-16].mean(0));res['menu_fixed'].append(T[v][:,F,am].mean(0))
 ch[v]=dict(pick16=a16,pick8=a8,union=au,menu=KEYS[am])
 print(f"{v:26s} rho_prop {spearmanr(fp.mean((0,1)),yp).correlation:5.2f} | gen16 {yp[a16]*100:5.1f} {np.round(G[v]['proposals'][a16],2).tolist()} gen8 {yp[a8]*100:5.1f} best {yp.max()*100:5.1f} | union {(yp[au] if au<16 else ym[au-16])*100:5.1f}{'(gen)' if au<16 else '(menu)'} | menu {ym[am]*100:5.1f} {KEYS[am]}")
arr={k:np.array(v) for k,v in res.items()}
glob_=np.array([T[v][:,F,KEYS.index('sE22')].mean(0) for v in VA])
rng=np.random.default_rng(2026102696);M,Sn=8,16;si=rng.integers(Sn,size=(5000,M,Sn));sc=rng.integers(M,size=(5000,M));sh=rng.integers(Sn,size=(5000,M,Sn))
for k,a in arr.items():
 d=a-glob_;c1=np.percentile(d[np.arange(M)[None,:,None],si].mean((1,2)),[2.5,97.5]);c2=np.percentile(d[sc[:,:,None],sh].mean((1,2)),[2.5,97.5])
 print(f"{k:22s} {a.mean()*100:6.2f}  minus global {d.mean()*100:+6.2f} state [{c1[0]*100:+.2f},{c1[1]*100:+.2f}] scene->state [{c2[0]*100:+.2f},{c2[1]*100:+.2f}]")
json.dump(dict(choices=ch,rates={k:float(a.mean()) for k,a in arr.items()}),open('FIX_CRITIC_TOP1.json','w'),indent=1)
