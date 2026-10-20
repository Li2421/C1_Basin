from core import *
r=read(HERE/'results.json');rates=r['rates'];ARMS=['F10','F13','S300','S500']
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10});out=HERE/'figures';out.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(9,4));x=np.arange(4)
for g,m in enumerate(['shared','local','central']):
 vals=[rates[m+'/'+a]['success_lower'] for a in ARMS];ups=[rates[m+'/'+a]['success_upper'] for a in ARMS];ax.bar(x+(g-1)*.24,vals,.24,label=m);ax.vlines(x+(g-1)*.24,vals,ups,color='black')
ax.set(xticks=x,xticklabels=ARMS,ylabel='Success proportion (unknown lower; line to upper)',ylim=(0,1));ax.legend();fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'pipeline_success.{ext}',dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));keys=['shared/F10','shared/F13','shared/S300','shared/S500','rhr/B-RHR','rhr_f10/B-RHR+F10'];ax.bar(np.arange(6),[rates[k]['success_lower'] for k in keys]);ax.set(xticks=np.arange(6),xticklabels=['F10','F13','S300','S500','RHR','RHR+F10'],ylabel='Success proportion',ylim=(0,1));fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'baseline_comparison.{ext}',dpi=180)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4))
for g,m in enumerate(['shared','local','central']):
 for ax,key in zip(axs,['stall_episode_rate','mean_stall_total_seconds']):ax.bar(x+(g-1)*.24,[rates[m+'/'+a][key] for a in ARMS],.24,label=m);ax.set(xticks=x,xticklabels=ARMS)
axs[0].set_ylabel('Episodes with diagnostic stall >=2s');axs[1].set_ylabel('Mean total stalled robot-seconds');axs[0].legend();fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'stall_diagnostics.{ext}',dpi=180)
plt.close(fig)
