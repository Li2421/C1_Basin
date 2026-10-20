from core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=read(HERE/'FEWSHOT_RESULTS.json');fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
for ax,scene in zip(axes.flat,['toy_give_way','double_bottleneck','four_way_intersection','ring_exchange']):
 unit=r[scene];ax.set_title(scene.replace('_',' '))
 if 'curves' not in unit:ax.text(.5,.5,'Prospective evaluation unavailable\nFP32 reproduction gate failed',ha='center',va='center',transform=ax.transAxes);continue
 for method in ['D0','E','S']:
  values=unit['curves'][method];ks=[int(k) for k in values];ax.plot(ks,[values[str(k)]['mean_regret_pp'] for k in ks],'-o',label=method)
 ax.axhline(3,color='gray',linestyle='--',linewidth=1);ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],[1,2,4,8,16]);ax.grid(alpha=.25);ax.set_ylabel('Mean regret (pp)');ax.set_xlabel('Known calibration episodes per arm (7 arms)');ax.legend()
fig.tight_layout()
for ext in ['svg','png','pdf']:fig.savefig(HERE/('fewshot_curves.'+ext),dpi=160)
plt.close(fig)
