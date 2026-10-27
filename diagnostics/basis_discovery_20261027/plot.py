"""Scientific overview, including all15 families and unfavorable fixed policies."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;s=json.load(open(OUT/'summary.json'));names=list(s['primary_single'])
fig,ax=plt.subplots(1,2,figsize=(13,6))
y=np.arange(len(names));ax[0].barh(y-.18,[s['primary_single'][n]['rescued']/14 for n in names],height=.35,label='Initial-state rescue /14')
ax[0].barh(y+.18,[s['restart'][n]['rescued']/6 for n in names],height=.35,label='Failed-state recovery /6')
ax[0].set(yticks=y,yticklabels=names,xlim=(0,1.1),xlabel='Finite single-column Oracle, main4 configs');ax[0].invert_yaxis();ax[0].legend(fontsize=8)
labels=['B3_frozen','safety','RHR_B','compressed_fixed','compressed_release','T5_fixed'];m=s['fixed']['confirm']
values=[m[n]['success']/16 for n in labels];lo=[v-m[n]['success_ci'][0]for v,n in zip(values,labels)];hi=[m[n]['success_ci'][1]-v for v,n in zip(values,labels)]
ax[1].barh(labels,values,xerr=np.array([lo,hi]),capsize=3);ax[1].set(xlim=(0,1.05),xlabel='Fresh confirmatory fixed-policy success /16\n95% descriptive CI,8 physical-state blocks')
for a in ax:a.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'results.png',dpi=180);fig.savefig(OUT/'results.pdf')
