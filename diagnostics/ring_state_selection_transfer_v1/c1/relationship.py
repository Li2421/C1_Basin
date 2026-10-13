"""Controller/state/eta independent-half ANOVA on completed C1 labels."""
import sys,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.ring_state_selection_transfer_v1.study import read,write
HERE=Path(__file__).resolve().parent

def effects(q):
 # axes controller,state,eta. Pure effects by successive marginalization and centering.
 out={}
 for mask in range(1,8):
  keep=[a for a in range(3) if mask&(1<<a)];v=q.mean(tuple(a for a in range(3) if a not in keep),keepdims=True) if len(keep)<3 else q.copy()
  for ax in keep:v=v-v.mean(ax,keepdims=True)
  out[mask]=v
 return out

def main():
 y=np.load(HERE/'labels.npz')['Y'][:5];lo=(y==1).astype(float);un=(y<0).astype(float);a=lo[:,:,:,:4].mean(-1);b=lo[:,:,:,4:].mean(-1);e,f=effects(a),effects(b);names={1:'controller',2:'state',3:'controller_state',4:'eta',5:'controller_eta',6:'state_eta',7:'controller_state_eta'};point={names[k]:float(np.mean(e[k]*f[k])) for k in e};rng=np.random.default_rng(2026101103);boot=[]
 for _ in range(5000):
  ix=rng.integers(32,size=32);e,f=effects(a[:,ix]),effects(b[:,ix]);boot.append([np.mean(e[k]*f[k]) for k in range(1,8)])
 qq=lo.mean(-1);uu=un.mean(-1)
 write(HERE/'RELATIONSHIP.json',dict(controller_names=list(read(HERE/'protocol.json')['kappa'])[:5],Q_eta_mean=qq.mean(1).tolist(),Q_eta_unknown_width=uu.mean(1).tolist(),Q_level=qq.mean((1,2)).tolist(),half_cross_product_ANOVA=point,ANOVA_state_bootstrap95={names[k]:np.quantile(np.array(boot)[:,k-1],[.025,.975]).tolist() for k in names},unknown_count=int(un.sum()),energy_scope='Lower endpoint estimand with unknown outcomes. Independence removes Bernoulli noise bias if labels complete; unknown-censored effects require bounds before interpreting as true-Q energy.',same_state_controller_intervention=True,causal_scope='Intervention on complete kappa-dependent FF program with fixed nativeFlowendpoint and safety, not isolated rotation-angle causality.'))
if __name__=='__main__':main()
