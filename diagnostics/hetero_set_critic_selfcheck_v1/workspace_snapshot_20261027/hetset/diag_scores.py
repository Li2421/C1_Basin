"""Recompute ranking-critic logits (same 10 seeds) for proposals and menu on validation inputs; save."""
import json,pickle,numpy as np,jax
import critic as CR,gen as GN
sp=json.load(open('SPLIT.json'));data=pickle.load(open('critic_data.pkl','rb'));G=json.load(open('GEN_FROZEN.json'));TR=sp['train'];VA=sp['val']
out={v:dict(prop=[],menu=[]) for v in VA}
for seed in range(10):
 net,params,nm=CR.train(data,TR,'list',seed);ap=jax.jit(lambda p,x:net.apply(p,x))
 for v in VA:
  for key,E in [('prop',np.array(G[v]['proposals'])),('menu',CR.MENU)]:
   out[v][key].append(np.array([[np.asarray(ap(params,GN.Xe(data[v]['c'],data[v]['h'][s],k,nm,E))) for s in range(24)] for k in range(3)]))
np.savez_compressed('diag_logits.npz',**{f'{v}__{k}':np.array(out[v][k]) for v in VA for k in ('prop','menu')})
print('saved')
