"""Plot the three distinct waiting outcomes from the fixed protocol."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/c1_progress_conditioned_wait_audit'
BASE=ROOT/'results/c1_multistep_objective_audit/traces'
fig,axes=plt.subplots(3,3,figsize=(14,9))
for col,(rid,a,title) in enumerate([(192,2,'ID192 a2: successful yielding'),(77,1,'ID77 a1: delayed success'),(208,3,'ID208 a3: eventual deadlock')]):
    d=np.load(OUT/'scores'/f'{rid:04d}_{a}.npz');z=np.load(BASE/f'{rid:04d}_20260916_{a}.npz')
    t=np.arange(100,300)*.05
    axes[0,col].plot(t,d['S'],label='old S',lw=1)
    axes[0,col].plot(t,d['Snew'],label='conditioned S',lw=1)
    axes[0,col].set(title=title,ylabel='low-speed penalty',ylim=(-.03,1.03))
    axes[0,col].legend(fontsize=8)
    axes[1,col].plot(t,d['C'],color='tab:green');axes[1,col].set(ylabel='C (remaining window to15s)',ylim=(-.03,1.03))
    dist=np.linalg.norm(z['positions_after']-z['goals'],axis=2)
    tt=(np.arange(len(dist))+1)*.05
    axes[2,col].plot(tt,dist[:,0],label='agent0');axes[2,col].plot(tt,dist[:,1],label='agent1')
    axes[2,col].axvspan(5,15,color='grey',alpha=.15)
    axes[2,col].set(ylabel='actual goal distance',xlabel='time (s)',xlim=(0,42.5))
    axes[2,col].legend(fontsize=8)
    for ax in axes[:,col]:ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'waiting_cases.png',dpi=150)
