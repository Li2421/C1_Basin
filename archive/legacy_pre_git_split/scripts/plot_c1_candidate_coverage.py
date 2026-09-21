"""Original versus selected expanded intervention, using executed trajectories."""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from single_integrator.environment import Config,GiveWayEnv
OUT=ROOT/'results/c1_candidate_coverage_expansion'
s=json.loads((OUT/'summary.json').read_text())
old=json.loads((ROOT/'results/c1_wait_horizon_sweep/choices.json').read_text())
walls=GiveWayEnv(Config(corridor_half_length=1.3)).walls
fig,axes=plt.subplots(3,2,figsize=(13,11))
for col,rid in enumerate([68,208]):
    action=next(r['action'] for r in old if r['rid']==rid and r['base_arm']=='PGtwo' and r['horizon']==25)
    selected=s[str(rid)]['selected'];cid=selected['cid']
    paths=[ROOT/'results/c1_multistep_objective_audit/traces'/f'{rid:04d}_20260916_{action}.npz',OUT/'traces'/f'{rid:04d}_{cid}.npz']
    for w in walls:axes[0,col].plot(w[:,0],w[:,1],color='black',lw=2)
    for path,label,color in zip(paths,['original selection','expanded selection'],['tab:orange','tab:blue']):
        z=np.load(path);x=z['positions_after'];t=(np.arange(len(x))+1)*.05
        for agent,style in [(0,'-'),(1,'--')]:
            axes[0,col].plot(x[:,agent,0],x[:,agent,1],color=color,ls=style,label=f'{label}, agent{agent}')
        dist=np.linalg.norm(x-z['goals'],axis=2)
        axes[1,col].plot(t,dist.max(1),color=color,label=label)
        speed=np.max(np.linalg.norm(z['applied'].reshape(-1,2,2),axis=2),axis=1)
        axes[2,col].plot(t,speed,color=color,label=label,lw=.8)
    axes[0,col].set(title=f'ID{rid}',xlabel='x',ylabel='y',ylim=(-.3,.75),xlim=(-1.4,1.4))
    axes[0,col].legend(fontsize=7)
    axes[1,col].set(ylabel='max goal distance',xlim=(0,42.5));axes[1,col].legend(fontsize=8)
    axes[2,col].set(ylabel='actual max agent speed',xlabel='time (s)',xlim=(0,42.5))
    for ax in axes[1:,col]:ax.axvline(25,color='grey',ls=':',label='prediction end');ax.axvspan(5,9,color='grey',alpha=.1)
    for ax in axes[:,col]:ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'selected_trajectories.png',dpi=150)
