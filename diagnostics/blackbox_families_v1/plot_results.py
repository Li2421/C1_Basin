"""Static scientific figure of the preregistered pilot-label cost curve."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=read(HERE/'results.json');names={'centralized_sampling':'Centralized random-shooting model predictive control','cross_entropy':'Cross-entropy method model predictive control','prioritized_astar':'Prioritized space-time A-star planning','deterministic':'Deterministic behavior-cloning policy','mixture':'Mixture-density behavior-cloning policy','diffusion':'Diffusion behavior-cloning policy','combined':'All five implementation-valid families'}
fig,ax=plt.subplots(figsize=(11,5.6));ks=np.array([2,4,8,16]);items=list(r['families'].items())+[('combined',r['combined'])]
for i,(family,unit) in enumerate(items):
 if not unit['methods']:continue
 lo=np.array([unit['methods']['pilot_'+str(k)]['mean_regret_pp'][0] for k in ks]);hi=np.array([unit['methods']['pilot_'+str(k)]['mean_regret_pp'][1] for k in ks]);color='black' if family=='combined' else plt.get_cmap('tab10')(i);ax.plot(ks,lo,'o-',color=color,lw=2.5 if family=='combined' else 1.4,label=names[family]);ax.fill_between(ks,lo,hi,color=color,alpha=.12)
ax.axhline(3,color='.5',ls='--',lw=1,label='Preregistered three-percentage-point criterion');ax.set_xscale('log',base=2);ax.set_xticks(ks,[str(x) for x in ks]);ax.set_xlabel('Pilot states per controller');ax.set_ylabel('Mean regret against cross-fitted full-menu oracle\n(percentage points)');ax.grid(alpha=.2);top=ax.secondary_xaxis('top',functions=(lambda x:2*x,lambda x:x/2));top.set_xticks(2*ks,[str(x) for x in 2*ks]);top.set_xlabel('Closed-loop labels per controller (two arms per pilot state)');ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.),frameon=False,fontsize=8);fig.text(.08,.015,'Shaded bands show unknown-outcome bounds, not sampling confidence intervals.',fontsize=8);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(HERE/'PILOT_LABEL_COST_CURVE.png',dpi=220,bbox_inches='tight');fig.savefig(HERE/'PILOT_LABEL_COST_CURVE.pdf',bbox_inches='tight');plt.close(fig)
