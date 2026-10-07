"""Static scientific figures, generated only from independent audit JSON."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def save(fig,name):
    fig.tight_layout(); fig.savefig(HERE/'figures'/name,dpi=160); plt.close(fig)

def main():
    (HERE/'figures').mkdir(exist_ok=True)
    fresh=read('fresh_replication.json')['comparisons']; x=np.arange(3)
    fig,ax=plt.subplots(figsize=(7,4))
    for arm,off,color in [('low',-.13,'#17806d'),('high',.13,'#b84135')]:
        q=np.array([r[arm]['probabilities']['deadlock'] for r in fresh])
        limits=np.array([r[arm]['exact_95_intervals']['deadlock'] for r in fresh])
        ax.errorbar(x+off,q,yerr=np.array([q-limits[:,0],limits[:,1]-q]),fmt='o',capsize=5,color=color,label=arm+' policy (32 seeds)')
    ax.set_xticks(x,[r['state_id'] for r in fresh]); ax.set(ylabel='Full-horizon deadlock probability',ylim=(-.04,1.08),title='Fresh GPU replication: exact 95% marginal intervals')
    ax.legend();ax.grid(alpha=.2);save(fig,'fresh_q_d.png')
    fig,axes=plt.subplots(1,3,figsize=(10,3.8),sharey=True)
    for ax,r in zip(axes,fresh):
        bottom=np.zeros(2)
        for event,color in [('deadlock','#b84135'),('success','#17806d'),('timeout','#d9a441'),('collision','#60558b')]:
            vals=np.array([r[a]['probabilities'][event] for a in ['low','high']])
            ax.bar(['low','high'],vals,bottom=bottom,color=color,label=event);bottom+=vals
        ax.set_title(r['state_id']);ax.set_ylim(0,1.05)
    axes[0].set_ylabel('Full-horizon outcome probability');axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1));save(fig,'outcomes.png')
    slices=read('nonsmoothness_audit.json')['slices']; chosen=['D1_pair231','D2_pair228','D8_pair225','S8g_pair229']
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True)
    for ax,sid in zip(axes.flat,chosen):
        r=next(s for s in slices if s['state_id']==sid)
        for field,off,label,col in [('all_seed_estimates',-.006,'all 4 (selected seed included)','#48639c'),('fresh_only_estimates',.006,'3 fresh seeds only','#b84135')]:
            vals=r[field]; q=np.array([s['probabilities']['deadlock'] for s in vals]); lim=np.array([s['exact_95_intervals']['deadlock'] for s in vals])
            ax.errorbar(np.array(r['phi_goal'])+off,q,yerr=[q-lim[:,0],lim[:,1]-q],fmt='o',capsize=3,color=col,label=label)
        ax.set_title(sid);ax.set(xlabel='goal gain (other gains zero)',ylabel='Q_D',ylim=(-.04,1.05));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,loc='center right');fig.suptitle('Historical coarse slices: points do not prove discontinuity');save(fig,'phi_slices.png')

if __name__=='__main__':main()
