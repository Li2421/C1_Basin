"""Publication-style cached physical trajectory and held-out PR figures."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'part2_closed_loop_sensitivity_validation_20261012'
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)

def readcsv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def draw_case(scene,uid,old,new):
    r=next(x for x in readcsv(PREV/'gain_loss_audit.csv') if x['scene']==scene and x['uid']==uid and x['eta_index']=='4'
           and x['seed']=='0' and x['old']==old and x['new']==new)
    loaded={}
    for ch,path in ((old,r['old_result']),(new,r['new_result'])):
        with np.load(Path(path).with_suffix('.npz')) as z:
            loaded[ch]={k:z[k].copy() for k in ('positions','actions','goals')}
        loaded[ch]['meta']=json.loads(Path(path).read_text())
    fig,(ax,err)=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    colors={old:'#2171b5',new:'#d94801'}
    n=loaded[old]['positions'].shape[1]
    if scene=='toy_give_way':
        ax.axhspan(-.2,.2,color='#f0f0f0',alpha=.5)
        ax.add_patch(plt.Rectangle((-.2,.2),.4,.48,facecolor='#bdbdbd',alpha=.24,edgecolor='black'))
    else:
        # Visual guide only: obstacle dimensions are in the frozen scenario config.
        ax.axhline(0,color='#ddd',lw=.5);ax.axvline(0,color='#ddd',lw=.5)
    for ch in (old,new):
        data=loaded[ch];p=data['positions'];g=data['goals']
        for i in range(n):
            ax.plot(p[:,i,0],p[:,i,1],color=colors[ch],alpha=.7,lw=1.15,
                    label=f'{ch} robot {i}' if i<2 else None)
            ax.plot(p[0,i,0],p[0,i,1],marker='o',ms=4,color=colors[ch])
            ax.plot(p[-1,i,0],p[-1,i,1],marker='s',ms=4,color=colors[ch])
        worst=np.linalg.norm(p-g[None],axis=2).max(axis=1)
        err.plot(np.arange(len(worst)),worst,color=colors[ch],lw=1.7,
                 label=f'{ch}: {data["meta"]["termination"]}, {len(p)-1} steps')
    for i,goal in enumerate(loaded[old]['goals']):ax.plot(*goal,marker='x',ms=7,color='black')
    ax.set_aspect('equal',adjustable='box');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.legend(fontsize=7,ncol=2)
    err.axhline(.08 if scene=='toy_give_way' else .09,color='black',ls='--',lw=.8,label='goal tolerance')
    err.set_xlabel('physical step');err.set_ylabel('worst robot goal error [m]');err.set_yscale('log')
    err.legend(fontsize=8)
    fig.suptitle(f'{scene} / {uid[-6:]} / registered eta[4] / seed 0: {old} vs {new}')
    fig.savefig(OUT/f'{scene}_eta4_seed0.svg');plt.close(fig)

def pr_curve(rows,key):
    y=np.asarray([int(r['target']) for r in rows],bool);s=np.asarray([float(r[key]) for r in rows])
    order=np.argsort(-s,kind='stable');y=y[order];s=s[order]
    cut=np.r_[np.flatnonzero(np.diff(s)!=0)+1,len(s)]
    tp=np.cumsum(y)[cut-1];precision=tp/cut;recall=tp/max(int(y.sum()),1)
    return np.r_[0,recall],np.r_[1,precision]

def draw_pr():
    rows=[r for r in readcsv(HERE/'heldout_prediction.csv') if r['split']=='heldout_test']
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for ax,scene,name in zip(axes,('toy_give_way','ring_exchange'),('Toy FT→FF loss','Ring TT→TF gain')):
        s=[r for r in rows if r['scene']==scene];prev=np.mean([int(r['target']) for r in s])
        ax.axhline(prev,color='#666',ls='--',label=f'prevalence {prev:.3f}')
        for key,label,c in [('condition0_score','frozen initial-action score','#2171b5'),
                            ('condition1_score','frozen early-response score','#d94801'),
                            ('first_action_l2','first-action magnitude','#31a354')]:
            q=[r for r in s if r[key] not in ('',None)]
            rec,pre=pr_curve(q,key);ax.step(rec,pre,where='post',label=label,color=c,lw=1.3)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='recall',ylabel='precision',title=name)
        ax.legend(fontsize=7)
    fig.savefig(OUT/'heldout_pr.svg');plt.close(fig)

def main():
    draw_case('toy_give_way','hard_confirm_toy_1_820001','FT','FF')
    draw_case('ring_exchange','hard_confirm_ring_1_920001','TT','TF')
    draw_pr();print('figures done')

if __name__=='__main__':main()
