"""Plot saved diagnostic evidence; no control changes or new rollouts."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.environment import Config, GiveWayEnv


def main():
    out = ROOT/'results/c1_diagnosis_trajectories'
    base = ROOT/'results/c1_dataset_extended_h120_seed0_baselines'
    corrected = ROOT/'results/c1_dataset_extended_h120_seed0_c1'
    config = json.loads((base/'config.json').read_text())
    env = GiveWayEnv(Config(**config['environment']))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for row, rid in enumerate((2,10)):
        with np.load(base/'mac_cbf'/f'rollout_{rid:04d}.npz') as z:
            safety = {k:z[k] for k in ('positions','goal_errors','initial_positions')}
        with np.load(corrected/f'rollout_{rid:04d}.npz') as z:
            c1 = {k:z[k] for k in ('positions','goal_errors','initial_positions','c1_correction','c1_applied_correction')}
        ax = axes[row,0]
        for a,b in env.walls:
            ax.plot([a[0],b[0]],[a[1],b[1]],color='black',lw=1)
        for agent, color in enumerate(('tab:blue','tab:orange')):
            for name, data, style in (('Safety',safety,'--'),('C1',c1,'-')):
                p = np.concatenate((data['initial_positions'][None], data['positions']))
                ax.plot(p[:,agent,0],p[:,agent,1],style,color=color,lw=1.5,label=f'{name} agent {agent}')
                ax.scatter(p[-1,agent,0],p[-1,agent,1],color=color,marker='x' if name=='C1' else 'o',s=30)
            ax.scatter(*env.goals[agent],marker='*',color=color,s=70)
        ax.set(title=f'Case {rid}: trajectories (x = C1 endpoint)',xlabel='x (m)',ylabel='y (m)',aspect='equal')
        if row==0:
            ax.legend(fontsize=7,loc='upper right')
        ax = axes[row,1]
        for name,data,color in (('Safety',safety,'tab:green'),('C1',c1,'tab:red')):
            t = np.arange(1,len(data['positions'])+1)*.05
            ax.plot(t,np.max(data['goal_errors'],axis=1),label=name,color=color)
        ax.axhline(.08,ls=':',color='black',label='goal tolerance')
        ax.set(title='Worst-agent goal distance',xlabel='Time (s)',ylabel='Distance (m)')
        ax.legend(fontsize=8)
        ax = axes[row,2]
        t = np.arange(1,len(c1['positions'])+1)*.05
        for key,label in (('c1_correction','Raw residual'),('c1_applied_correction','Executed deviation')):
            ax.plot(t,np.linalg.norm(c1[key],axis=1)*1000,label=label,alpha=.8,lw=.8)
        ax.set(title='C1 correction magnitude',xlabel='Time (s)',ylabel='Joint norm (mm/s)')
        ax.legend(fontsize=8)
    fig.savefig(out/'failure_mechanisms.png',dpi=180)
    plt.close(fig)
    training = json.loads((ROOT/'results/c1_diagnosis_training/report.json').read_text())
    history = json.loads((ROOT/'results/c1_dataset_audit_seed0/history.json').read_text())
    epsilon = json.loads((ROOT/'results/c1_dataset_audit_seed0/config.json').read_text())['training']['epsilon']
    rows = training['scales']
    fig, axes = plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    scales = [r['scale'] for r in rows]
    risks = [r['J_live'] for r in rows]
    losses = [r['J_def']+history[1]['lambda_used']*(r['J_live']-epsilon) for r in rows]
    for ax,values,title in zip(axes,(risks,losses),('Risk on the actual updated batch','Primal loss with lambda held fixed')):
        ax.plot(scales,values,'o-')
        ax.axhline(values[0],ls='--',color='gray',label='Before update')
        ax.set(title=title,xlabel='Fraction of actual parameter update')
        ax.legend(fontsize=8)
    fig.savefig(ROOT/'results/c1_diagnosis_training/update_replay.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
