"""Publication-exportable figures from audited saved metrics only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
read=lambda n:json.loads((HERE/n).read_text())
def save(fig,name):
 d=HERE/'figures';d.mkdir(exist_ok=True);fig.savefig(d/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(d/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def main():
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 p=[read('pilot_results.json'),read('pilot_revision_results.json'),read('broad_pilot_results.json'),read('horizon_pilot_results.json')['results']['400']];labels=['Native starts\nH850','Deeper bay\nH850','Broad starts\nH850','Broad starts\nH400'];fig,ax=plt.subplots(1,2,figsize=(10,3.4))
 for j,key in enumerate(['state_fraction','oracle_adaptive_gain']):
  v=np.array([x[key] for x in p])*100;ci=np.array([x[key+'_ci95'] for x in p])*100;ax[j].errorbar(range(4),v,yerr=np.stack([v-ci[:,0],ci[:,1]-v]),fmt='o',capsize=4,color='#2364aa');ax[j].axhline(20 if j==0 else 5,ls='--',color='#d35400',label='Preregistered point threshold');ax[j].axhline(0,color='.7',lw=.6);ax[j].set_xticks(range(4),labels);ax[j].set_ylabel('State-related signal (%)' if j==0 else 'Crossfit adaptive gain (pp)');ax[j].legend(fontsize=8)
 fig.suptitle('State-dependent Q does not establish state-dependent eta selection');fig.tight_layout();save(fig,'pilot_gate')
 r=read('legacy_diagnostic_results.json')['results'];names=['ENT_MANY','ENT_FEW','ENT_FEW_x2','ENT_ALL','ENT_raw_MANY','ENT_mech_MANY','RANK_MANY'];fig,ax=plt.subplots(1,2,figsize=(11,3.7))
 for i,n in enumerate(names):
  q=r[n]['interpolation']['ensemble'];a=q['state_eta'];ax[0].scatter(i,a['corr'],color='#2364aa');ci=a['corr_ci95'];ax[0].vlines(i,*ci,color='#2364aa');ax[1].scatter(i,100*q['success'],color='#2364aa');ax[1].scatter(i,100*q['per_controller_fixed'],marker='_',color='#d35400')
 for a in ax:a.set_xticks(range(len(names)),[n.replace('ENT_','') for n in names],rotation=30,ha='right')
 ax[0].set_ylabel('Held state x eta correlation');ax[1].set_ylabel('Selected closed-loop success (%)');ax[1].set_ylim(95,101);ax[1].plot([],[],'.',color='#2364aa',label='Learned selection');ax[1].plot([],[],'_',color='#d35400',label='Source per-C fixed eta');ax[1].legend(fontsize=8);fig.suptitle('Legacy Toy widths: prediction varies; selection has almost no headroom');fig.tight_layout();save(fig,'legacy_transfer')
 c=read('calibration_results.json');fig,axes=plt.subplots(1,4,figsize=(13,3.3),sharey=True)
 for ax,name in zip(axes,['toy_give_way','double_bottleneck','four_way_intersection','ring_exchange']):
  rr=c[name]['results']['ensemble'];xx=[0,1,2,4]
  for method,color,label in [('prior_only','#d35400','Target prior'),('prior_conditional','#2364aa','Prior + conditional')]:
   y=[rr[str(k)]['methods'][method]['success']['mean']*100 for k in xx];ci=np.array([rr[str(k)]['methods'][method]['success']['state_ci95'] for k in xx])*100;ax.plot(xx,y,'o-',color=color,label=label);ax.fill_between(xx,ci[:,0],ci[:,1],color=color,alpha=.12)
  ax.axhline(rr['0']['methods']['zero_shot']['success']['mean']*100,ls='--',color='.4',label='Zero-shot model');ax.set_title({'toy_give_way':'Toy','double_bottleneck':'Double bottleneck','four_way_intersection':'Four-way','ring_exchange':'Ring'}[name]);ax.set_xticks(xx);ax.set_xlabel('Target calibration states')
 axes[0].set_ylabel('Success lower bound (%)');axes[-1].legend(fontsize=7,loc='lower right');fig.tight_layout();save(fig,'calibration')
 f=read('factor_results.json')['results'];regs=['WIDTH','EXTRAP','COMBINATION','FAMILY','FAMILY_REVERSE','COUNT','TIME'];arms=['ENT','ENT_raw','ENT_mech','MECH'];vals=np.array([[f[f'{a}_{reg}']['all']['ensemble']['success'] for reg in regs] for a in arms]);fig,ax=plt.subplots(figsize=(10,3.4));im=ax.imshow(vals,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(len(regs)),regs,rotation=20,ha='right');ax.set_yticks(range(len(arms)),arms)
 for i in range(len(arms)):
  for j in range(len(regs)):ax.text(j,i,f'{vals[i,j]*100:.1f}%',ha='center',va='center',color='white' if vals[i,j]<.6 else 'black')
 fig.colorbar(im,ax=ax,label='Selected success');ax.set_title('Independent-factor transfer: given-controller eta selection (Q2)');fig.tight_layout();save(fig,'factor_transfer')
if __name__=='__main__':main()
