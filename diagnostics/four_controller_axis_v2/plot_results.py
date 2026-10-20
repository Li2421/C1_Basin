"""Static numerical figures from frozen probe and merged rollout tables."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'diagnostics/controller_eta_switch_time_v1/vendor'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read(p):return json.loads(Path(p).read_text())
def main():
    out=HERE/'figures';out.mkdir(exist_ok=True)
    fixed=read(HERE/'FIXED_RESULTS.json');names=[r['controller'] for r in fixed['table']]
    rates=np.asarray([r['success_rates'] for r in fixed['table']])*100
    labels=[f'{i+1:02d} {n.removeprefix("base_u_").replace("_macflow","")[:38]}' for i,n in enumerate(names)]
    fig,ax=plt.subplots(figsize=(7.5,10));im=ax.imshow(rates,aspect='auto',vmin=0,vmax=100,cmap='viridis')
    ax.set_xticks(range(3),fixed['arms']);ax.set_yticks(range(25),labels,fontsize=6)
    for i in range(25):
        for j in range(3):ax.text(j,i,f'{rates[i,j]:.0f}',ha='center',va='center',fontsize=6,
                                   color='white' if rates[i,j]<60 else 'black')
    fig.colorbar(im,ax=ax,label='success among known seeds (%)');fig.tight_layout();fig.savefig(out/'fixed_rates.png',dpi=170);plt.close(fig)
    z=np.load(HERE/'R5_FROZEN.npz');scores=z['scores'];fig,ax=plt.subplots(figsize=(7,5))
    for i,(x,y) in enumerate(scores):
        ax.scatter(x,y,s=28,color='#0072B2' if fixed['table'][i]['flavor']=='best' else '#D55E00')
        ax.annotate(str(i+1),(x,y),fontsize=7,xytext=(3,3),textcoords='offset points')
    ax.set_xlabel('R5 PCA-1');ax.set_ylabel('R5 PCA-2');ax.set_title('Frozen unlabeled 200-step probe')
    fig.tight_layout();fig.savefig(out/'r5_pca.png',dpi=170);plt.close(fig)
    p=HERE/'SCHEDULE_RESULTS.json'
    if p.exists():
        sched=read(p);effects=sched['effects'];arms=sched['all_arms'][3:]
        E=np.asarray([[next(r['E_pp'] for r in effects if r['controller']==name and r['arm']==a) for a in arms] for name in names])
        L=np.asarray([[next(r['L_pp'] for r in effects if r['controller']==name and r['arm']==a) for a in arms] for name in names])
        fig,axes=plt.subplots(1,2,figsize=(13,10),sharey=True)
        for ax,arr,title in zip(axes,(E,L),('E = schedule − late fixed','L = schedule − early fixed')):
            image=ax.imshow(arr,aspect='auto',vmin=-100,vmax=100,cmap='coolwarm');ax.set_title(title)
            ax.set_xticks(range(5),arms,rotation=50,ha='right',fontsize=7)
        axes[0].set_yticks(range(25),labels,fontsize=6);fig.colorbar(image,ax=axes,label='percentage points',fraction=.026)
        fig.savefig(out/'schedule_effects.png',dpi=170,bbox_inches='tight');plt.close(fig)
    print('figures written',flush=True)
if __name__=='__main__':main()
