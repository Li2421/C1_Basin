"""Plot only frozen simulation/training evidence; never change execution."""
from core import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_utils import source_inventory

def main():
    rows=[];fig,axes=plt.subplots(2,5,figsize=(18,7),constrained_layout=True)
    for ax,i in zip(axes.flat,read(HERE/'GLOBAL_INDICES.json')):
        r=read(HERE/'TEACHERS.json')[i];folder=HERE/'students'/r['name'];history=read(folder/'history.json');best=read(folder/'COMPLETE.json')['best_step'];rows.extend([dict(controller=r['name'],**v) for v in history]);ax.plot([x['step'] for x in history],[x['dev_mse'] for x in history],label='fixed dev');ax.plot([x['step'] for x in history if x['train_mse'] is not None],[x['train_mse'] for x in history if x['train_mse'] is not None],label='train');ax.axvline(best,color='k',linestyle=':',linewidth=.8);ax.set_yscale('log');ax.set_title(r['name'].replace('base_u_','').replace('_macflow',''),fontsize=9);ax.set_xlabel('Training step');ax.set_ylabel('MSE (m/s)^2')
    axes.flat[0].legend();fig.savefig(HERE/'training_curves.png',dpi=160);fig.savefig(HERE/'training_curves.pdf');plt.close(fig)
    with (HERE/'training_curves.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    eq=read(HERE/'EQUIVARIANCE.json')['controllers'];fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True);ax.bar(range(10),[x['relative_rms']*100 for x in eq]);ax.set_xticks(range(10),[x['teacher'].replace('base_u_','').replace('_macflow','') for x in eq],rotation=50,ha='right');ax.set_ylabel('Relative equivariance RMS error (%)');fig.savefig(HERE/'equivariance.png',dpi=160);plt.close(fig);source_inventory('imports_plots.json')
if __name__=='__main__':main()
