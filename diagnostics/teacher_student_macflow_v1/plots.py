from core import *
from audit_utils import source_inventory
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=HERE/'figures';p.mkdir(exist_ok=True);fig,axes=plt.subplots(4,4,figsize=(16,13),constrained_layout=True);summary=[]
for ax,r in zip(axes.flat,read(HERE/'TEACHERS.json')):
    folder=HERE/'students'/r['name'];h=read(folder/'history.json');c=read(folder/'COMPLETE.json');ax.plot([x['step'] for x in h],[x['dev_mse'] for x in h],label='fixed dev MSE');ax.plot([x['step'] for x in h if x['train_mse'] is not None],[x['train_mse'] for x in h if x['train_mse'] is not None],label='train MSE',alpha=.7);ax.axvline(c['best_step'],color='tab:red',linestyle=':',label='selected');ax.set_yscale('log');ax.set_title(r['name'].replace('base_u_','').replace('_macflow',''),fontsize=9);ax.set_xlabel('update');ax.set_ylabel('MSE');ax.grid(alpha=.15)
    for row in h:summary.append(dict(controller=r['name'],**row))
for ax in axes.flat[14:]:ax.set_visible(False)
axes.flat[0].legend(fontsize=7);fig.savefig(p/'training_curves.png',dpi=160);fig.savefig(p/'training_curves.svg');plt.close(fig)
with (HERE/'training_curves.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=['controller','step','dev_mse','train_mse','elapsed']);w.writeheader();w.writerows(summary)
if (HERE/'results.json').exists() and 'student_table' in read(HERE/'results.json'):
    q=read(HERE/'results.json');arms=['nominal only','safety baseline','F10','F13','S300','S500','B-RHR'];v=[q['student_table'][a]['success'] for a in arms];y=np.array([r['mean'] for r in v])*100;err=np.array([[r['mean']-r['CI95'][0],r['CI95'][1]-r['mean']] for r in v]).T*100;fig,ax=plt.subplots(figsize=(10,4));ax.bar(np.arange(7),y,yerr=err,capsize=3);ax.set_xticks(np.arange(7),arms,rotation=15);ax.set_ylabel('Success (%)');ax.set_ylim(0,105);fig.tight_layout();fig.savefig(p/'student_success.png',dpi=180);plt.close(fig)
source_inventory('imports_plots.json')
print('Plots written',flush=True)
