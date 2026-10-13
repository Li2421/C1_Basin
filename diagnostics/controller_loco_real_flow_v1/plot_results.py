"""Standalone scientific figure from saved paired confidence intervals."""
import json
import os
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'plot_config'))
sys.path.insert(0,str(BASE/'plot_vendor'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def main():
    a=read(BASE/'STAGE_A_RESULTS_FROZEN.json')['primary']['contrasts']
    b=read(BASE/'stage_b/results.json')['primary']['contrasts']
    c=read(BASE/'stage_c/results.json')['contrasts']
    prior=read(BASE/'controller_prior/results.json')
    arms=['R3','R4_action','R4_openloop','R3_R4_action'];labels=['R3','R4 action','R4 open-loop','R3 + R4 action']
    fig,axs=plt.subplots(4,3,figsize=(13,10),sharey=True)
    for col,(stage,contrasts,title) in enumerate([('stage_a',a,'Stage A: 6 sources'),('stage_b',b,'Stage B: 13 sources / development'),('stage_c',c,'Stage C: frozen B / new states')]):
        for row,key in enumerate(['src','R0','mismatch','state_prior']):
            ax=axs[row,col];ax.axvline(0,color='0.5',lw=1,ls='--')
            entries=prior[stage]['contrasts'] if key=='state_prior' else [r for r in contrasts if r['comparison']==key]
            for r in entries:
                arm=r['arm'];y=arms.index(arm)
                value=r['state_selection_minus_learned_controller_prior_lower'] if key=='state_prior' else r['mean']
                lo,hi=r['ci95_simultaneous']
                ax.errorbar(value*100,y,xerr=np.array([[max(0,value-lo)*100],[max(0,hi-value)*100]]),fmt='o',capsize=3,color='#17659b' if lo>0 else '#666666')
            ax.set_yticks(range(4),labels);ax.set_ylim(3.6,-.6);ax.grid(axis='x',alpha=.2)
            if row==0:ax.set_title(title,fontsize=10)
            ax.set_xlabel(['Gain over source fixed eta (pp)','Gain over R0 (pp)','Matched - mismatched (pp)','State selector - learned prior (pp)'][row],fontsize=9)
    fig.suptitle('Controller-conditioned selection: conservative paired gains and simultaneous 95% CIs',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97))
    dest=BASE/'figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'controller_selection_gains.png',dpi=180)
    fig.savefig(dest/'controller_selection_gains.pdf')
    plt.close(fig)

if __name__=='__main__':main()
