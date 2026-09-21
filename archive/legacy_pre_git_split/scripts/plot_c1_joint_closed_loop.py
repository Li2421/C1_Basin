"""Plot recorded audit arrays; can use system Python with numpy/matplotlib."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/c1_audit_mpl')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/c1_joint_closed_loop_audit'

def main():
    rows=json.loads((OUT/'episodes.json').read_text())
    arms=['baseline','descent_002','descent_010','random_010']
    for rid in [2,52]:
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        for arm in arms:
            z=np.load(OUT/'traces'/f'{rid:04d}_{arm}.npz');t=(np.arange(len(z['risk']))+1)*float(z['dt'])
            r=next(r for r in rows if r['rid']==rid and r['arm']==arm)
            label=f"{arm} / {r['outcome']}"
            axes[0,0].plot(t,z['risk'],label=label,lw=1,alpha=.85)
            axes[0,1].plot(t,np.max(np.linalg.norm(z['applied'].reshape(-1,2,2),axis=-1),axis=1),lw=1)
            axes[1,0].plot(t,np.max(z['goal_errors'],axis=1),lw=1)
            axes[1,1].plot(t,z['stuck_timer'],lw=1)
        axes[0,0].set_title('Local risk');axes[0,0].legend(fontsize=7)
        axes[0,1].set_title('Maximum agent speed');axes[0,1].axhline(.025,color='gray',ls=':')
        axes[1,0].set_title('Maximum remaining goal distance')
        axes[1,1].set_title('Deadlock hold timer');axes[1,1].axhline(5,color='gray',ls=':')
        for ax in axes.flat:ax.set_xlabel('Time [s]');ax.grid(alpha=.2)
        fig.suptitle(f'Initial state {rid}: same frozen policy and random-number stream')
        fig.tight_layout();fig.savefig(OUT/f'closed_loop_{rid:04d}.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
