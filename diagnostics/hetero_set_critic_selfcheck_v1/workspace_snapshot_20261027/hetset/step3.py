"""After batch1: (1) val eval of ensemble generator + retrained critic; (2) test-scene eligibility (same rule);
(3) freeze ALL test choices (global, knn1-3, menu critic, ens gen + retrained critic 16/8, union) before any test fresh-state label exists."""
import json,pickle,subprocess,numpy as np,jax
import desc,critic as CR,critic2 as C2
from tab import KEYS
sp=json.load(open('SPLIT.json'));data=pickle.load(open('critic_data.pkl','rb'));TR=sp['train'];VA=sp['val']
TE_all={s['name']:s for s in json.load(open('scenes_test.json'))}
# --- eligibility on test gate (states 0-7, 49 menu): mean in [.1,.9], best>=.3 ; first 3 eligible per cluster in listed order
gate={}
for l in open('test_gate.jsonl'):
 d=json.loads(l);t=gate.setdefault(d['scene'],np.full((3,8,49),np.nan));t[d['controller'],d['state'],KEYS.index(d['key'][5:])]=d['success']
TE=[];per={}
for n,s in TE_all.items():
 q=gate[n].mean((0,1));ok=np.isfinite(gate[n]).all() and .1<=q.mean()<=.9 and q.max()>=.3
 if ok and per.get(s['cluster'],0)<3:TE.append(n);per[s['cluster']]=per.get(s['cluster'],0)+1
 print(f"test candidate {n:28s} mean {q.mean():.2f} best {KEYS[int(q.argmax())]} {q.max():.2f} -> {'TEST' if n in TE else 'out'}")
# --- test inputs
for n in TE_all:
 s=dict(TE_all[n])
 if s['cluster']=='obst':s['min_opening_over_diameter']=desc.literature_opening('lit_swap8_obstacles',s['walls'],s['config']['agent_radius'])
 s['features']=desc.features(s);h=desc.state_h(s);data[n]=dict(t=np.full((3,24,49),np.nan),c=desc.descriptor(s),h=h,h_pred=h)
pickle.dump(data,open('critic_data_with_test.pkl','wb'))
# --- ensemble generator proposals for val+test (trained on all train scenes)
json.dump(VA+TE,open('ENS_TARGETS.json','w'))
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
p=subprocess.run([PY,'gen_ens.py','all','critic_data_with_test.pkl'],capture_output=True,text=True);P=json.loads([l for l in p.stdout.splitlines() if l.startswith('PROPS')][0][6:])
PV=json.load(open('GEN_ENS_VAL_PROPOSALS.json'));print('val proposals reproduce:',max(np.abs(np.array(P[v])-np.array(PV[v]['proposals'])).max() for v in VA))
# --- retrained critic (menu + LOSO ensemble proposal labels on training scenes)
PT=json.load(open('GEN_ENS_LOSO_TRAIN_PROPOSALS.json'));tp={n:np.full((3,24,16),np.nan) for n in TR}
for l in open('ens_train.jsonl'):d=json.loads(l);tp[d['scene']][d['controller'],d['state'],int(d['key'][1:])]=d['success']
nm=CR.norm_of(data,TR)
scenes={n:dict(E=np.concatenate([CR.MENU,np.array(PT[n])]),t=np.concatenate([data[n]['t'],tp[n]],-1),c=data[n]['c'],h=data[n]['h']) for n in TR}
L={v:dict(p=[],m=[]) for v in VA+TE}
for seed in range(10):
 net,params=C2.train(scenes,nm,seed);ap=jax.jit(lambda p,x:net.apply(p,x))
 for v in VA+TE:
  for key,E in [('p',np.array(P[v])),('m',CR.MENU)]:L[v][key].append(np.array([[np.asarray(ap(params,C2.Xe(data[v]['c'],data[v]['h'][s],k,nm,E))) for s in range(24)] for k in range(3)]))
# --- menu-only ranking critic (as validated earlier): 10 seeds, predictions for test
Lm={v:[] for v in TE}
for seed in range(10):
 net,params,nmc=CR.train(data,TR,'list',seed)
 for v in TE:Lm[v].append(CR.predict(net,params,nmc,data[v]['c'],data[v]['h']))
def soft(f):q=np.exp(f-f.max(-1,keepdims=True));return (q/q.sum(-1,keepdims=True)).mean((0,1))
curve={n:np.nanmean(data[n]['t'][:,:16],(0,1)) for n in TR};C=np.array([data[n]['c'] for n in TR]);sig=C.std(0);sig[sig<1e-9]=1
G=int(np.mean([curve[n] for n in TR],0).argmax())
def knn(c,k):d=np.sqrt((((C-c)/sig)**2).sum(1));return int(np.mean([curve[TR[i]] for i in np.argsort(d,kind='stable')[:k]],0).argmax())
F={}
for v in VA+TE:
 fp=np.mean(L[v]['p'],0);fm=np.mean(L[v]['m'],0);fu=np.concatenate([fp,fm],-1);u=int(soft(fu).argmax())
 F[v]=dict(proposals=P[v],ens16=int(soft(fp).argmax()),ens8=int(soft(fp[...,:8]).argmax()),union=('p',u) if u<16 else ('menu',KEYS[u-16]),
  global_=KEYS[G],knn1=KEYS[knn(data[v]['c'],1)],knn2=KEYS[knn(data[v]['c'],2)],knn3=KEYS[knn(data[v]['c'],3)],
  menu_critic=KEYS[int(soft(np.mean(Lm[v],0)).argmax())] if v in TE else None)
json.dump(dict(test=TE,val=VA,choices=F,frozen_before_test_labels=True),open('FINAL_FROZEN.json','w'),indent=1)
print('frozen; test scenes',len(TE),TE)
