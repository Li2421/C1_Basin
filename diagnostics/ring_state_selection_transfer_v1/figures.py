"""Standalone scientific figures, no interactive reporting dependency."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
read=lambda f:json.loads((P/f).read_text())
def save(fig,name):
 fig.tight_layout();fig.savefig(P/(name+'.png'),dpi=180);fig.savefig(P/(name+'.pdf'));plt.close(fig)
def main():
 r=read('N1_RESULTS.json');fig,ax=plt.subplots(figsize=(7,4))
 for arm,label in [('scalar','Scalar Q'),('heads','16-head MLP'),('knn','Physical kNN')]:
  rows=[x for x in r['summary'] if x['arm']==arm];xx=[x['n'] for x in rows];yy=np.array([x['gain'] for x in rows])*100;ci=np.array([x['gain_ci95'] for x in rows])*100
  ax.errorbar(xx,yy,yerr=[yy-ci[:,0],ci[:,1]-yy],label=label,marker='o',capsize=3)
 ax.axhline(0,color='black',ls='--',lw=1);ax.set_xscale('log',base=2);ax.set_xticks([32,64,128,256,512],labels=['32','64','128','256','512']);ax.set_xlabel('Training states (K16, Q4)');ax.set_ylabel('Gain vs fixed eta13 (percentage points)');ax.set_title('Ring DEVELOPMENT: paired state-bootstrap 95% CI');ax.legend();save(fig,'learning_curve')
 c0=read('C0_FF_TT_FOCUS.json');a=np.load(P/'controller_panels'/(c0['panel']+'.npz'))['Y'];q=(a==1).mean(-1);u=(a<0).mean(-1);fig,ax=plt.subplots(figsize=(7,3.7))
 for i,name in enumerate(['Native FF','TT']):ax.plot(range(16),q[i].mean(0),marker='o',label=name);ax.fill_between(range(16),q[i].mean(0),(q[i]+u[i]).mean(0),alpha=.15)
 ax.set_xlabel('Original eta bank index');ax.set_ylabel('Closed-loop success lower bound');ax.set_title('Same 24 Ring states x K16 x Q16 (shading: unknown bounds)');ax.legend();save(fig,'controller_eta_curves')
 if (P/'c1/RESULTS.json').exists():
  c=read('c1/RESULTS.json');rr=c['primary'][:5];fig,ax=plt.subplots(figsize=(7,4));x=[v['kappa'] for v in rr]
  for key,label in [('fixed_crossfit_success','Cross-fitted fixed eta'),('oracle_crossfit_success_lower','Cross-fitted state oracle'),('hindsight_best_fixed_success','Hindsight best fixed')]:ax.plot(x,[100*v[key] for v in rr],marker='o',label=label)
  ax.set_xlabel('FF matrix parameter kappa (not effective response angle)');ax.set_ylabel('Success (%)');ax.set_title('C1: same native Flow endpoint, geometry, states and K16');ax.legend();save(fig,'controller_family_pilot')
if __name__=='__main__':main()
