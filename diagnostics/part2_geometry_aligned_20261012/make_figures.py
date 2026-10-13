"""Static figures from saved analysis CSVs; no controller or simulator calls."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent;FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
def rows(name):
    with (HERE/name).open(newline='') as f:return list(csv.DictReader(f))

def main():
    colors={'gain':'#2b8cbe','loss':'#d95f0e','both_success':'#31a354','both_failure':'#777777'}
    r=rows('ring_directional_mechanism.csv');t=rows('toy_yielding_mechanism.csv')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for outcome in ('both_failure','both_success','gain','loss'):
        z=[x for x in r if x['b15_outcome']==outcome]
        ax[0].scatter([float(x['new_pre_angular_abs'])-float(x['old_pre_angular_abs']) for x in z],
                      [float(x['projection_angular_change_advantage']) for x in z],
                      s=13,alpha=.65,c=colors[outcome],label=f'{outcome} ({len(z)})')
        z=[x for x in t if x['b15_outcome']==outcome]
        ax[1].scatter([float(x['goal_mean_difference']) for x in z],
                      [float(x['lateral_asymmetry_difference']) for x in z],
                      s=13,alpha=.65,c=colors[outcome],label=f'{outcome} ({len(z)})')
    ax[0].axvline(0,color='0.6',lw=.7);ax[0].axhline(0,color='0.6',lw=.7)
    ax[0].set(xlabel='TF−TT before terminal projection: absolute angular speed',
              ylabel='Relative terminal projection change',title='Ring: projection retention is common')
    ax[1].axvline(0,color='0.6',lw=.7);ax[1].axhline(0,color='0.6',lw=.7)
    ax[1].set(xlabel='FF−FT goal-aligned mean velocity',
              ylabel='FF−FT relative lateral speed magnitude',title='Toy: geometric features overlap')
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(FIG/'geometry_action_overlap.svg');plt.close(fig)

    s=rows('flow_substep_decomposition.csv')
    fig,ax=plt.subplots(1,2,figsize=(9,3.7))
    for i,scene in enumerate(('toy_give_way','ring_exchange')):
        z=[x for x in s if x['scene']==scene and x['uid'].endswith('001') and x['eta_index']=='4']
        for k,label in [('cumulative_direct','direct input'),('cumulative_feedback','Flow feedback'),
                        ('cumulative_projection','field projection'),('path_directional_difference','actual path difference')]:
            ax[i].plot([int(x['substep']) for x in z],[float(x[k]) for x in z],marker='o',ms=2.5,label=label)
        ax[i].axhline(0,color='0.65',lw=.7)
        ax[i].set(xlabel='Flow substep',ylabel='directional action units',
                  title=('Toy 820001 η4' if i==0 else 'Ring 920001 η4'))
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(FIG/'substep_directional_example.svg');plt.close(fig)
    print('figures',len(list(FIG.glob('*.svg'))))

if __name__=='__main__':main()
