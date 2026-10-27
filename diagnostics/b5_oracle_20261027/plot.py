"""Standalone shareable plots; reads completed summary only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
s=json.load(open(OUT/'summary.json'));curve=s['budget_curve'];ind=s['independent']
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
for name in ['B3']+list(s['oracle']['fixed']):
    r=[x for x in curve if x['method']==name]
    ax[0].plot([x['budget']for x in r],[x['solved']/64 for x in r],marker='.',label=name)
ax[0].set(xlabel='Candidate budget (staged search)',ylabel='Solved development cases / 64',ylim=(.4,1.02))
ax[0].legend(fontsize=7,ncol=2);ax[0].grid(alpha=.2)
names=['safety','B3','T_only','add_goal','blend_goal','add_cluster','blend_cluster','RHR_A','RHR_B']
display=[{'blend_cluster':'GRT-cluster (Y=0)', 'T_only':'GPR + T-goal (Y=0)'}.get(n,n)for n in names]
ax[1].barh(display,[ind[n]['rate']for n in names]);ax[1].set(xlabel='Frozen independent test success / 32',xlim=(0,1));ax[1].grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'results.png',dpi=180);fig.savefig(OUT/'results.pdf')
