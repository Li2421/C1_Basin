from load import *;import sys,glob
i29=KEYS.index('sE29')
def choose(pref,kind):
 fs=sorted(glob.glob(f'out/{pref}_*.npz'));L={v:np.mean([np.load(f)[v] for f in fs],0) for v in VAL}
 out={}
 for v,f in L.items():
  if kind=='bce':sc=(1/(1+np.exp(-f))).mean((0,1))
  else:
   p=np.exp(f-f.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);sc=p.mean((0,1))
  out[v]=int(sc.argmax())
 return out,len(fs)
for pn in ['all','filt']:
 for kind in ['bce','list','hybrid']:
  ch,k=choose(f'full_{pn}_{kind}',kind);r=[val_y(v)[:,:,ch[v]].mean() for v in VAL]
  print(f"{pn:5s} {kind:7s} seeds{k} mean {np.mean(r)*100:.2f}  ",' '.join(f"{v[:12]}:{KEYS[ch[v]]}={x*100:.0f}" for v,x in zip(VAL,r)))
