"""Plot saved gradient audit with system Python/Matplotlib; no model replay."""
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/c1-gradient-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/c1_gradient_paths_guarded'
r=json.loads((out/'report.json').read_text())
with np.load(out/'trace.npz') as data:
    m=data['margins']; mask=data['episode_mask'][...,None]&np.isfinite(m)
    values=m[mask]
fig,axes=plt.subplots(1,3,figsize=(14,4))
axes[0].hist(values,bins=75,density=True,color='steelblue')
axes[0].axvspan(.1*(1-np.log(9)/8),.1*(1+np.log(9)/8),color='orange',alpha=.4,label='cone risk 10–90%')
axes[0].set(xlabel='Signed cone margin',ylabel='Density',title='Guarded model: two calibration episodes')
axes[0].legend(fontsize=8)
names=['activity','cone','task','total']
axes[1].bar(names,[r['paths'][k]['directional_derivative'] for k in names],color=['orange','steelblue','green','grey'])
axes[1].axhline(0,color='black',lw=.7)
axes[1].set(ylabel='d mean R / d scale',title='Full BPTT along optimizer displacement')
fd=r['finite_differences']
axes[2].semilogx([x['step'] for x in fd],[x['derivative'] for x in fd],'o-',label='Central finite difference')
axes[2].axhline(r['paths']['total']['directional_derivative'],ls='--',color='black',label='Selected-branch autodiff')
axes[2].set(xlabel='Perturbation scale',ylabel='d mean R / d scale',title='Finite-step sensitivity')
axes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out/'gradient_diagnosis.png',dpi=180)
fig.savefig(out/'gradient_diagnosis.pdf')
