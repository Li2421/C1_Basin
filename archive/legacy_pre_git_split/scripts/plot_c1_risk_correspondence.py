"""Export risk/outcome correspondence from a complete frozen test only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS=('success','safe_deadlock','stalled_deadlock','other_timeout')
COLORS=('#329a64','#a92d36','#e79b38','#7e8994')
UPPER=np.array([.01,.05,.1,.25,.5,1.,2.,3.,np.inf])
BIN_NAMES=['0','(0,.01]','(.01,.05]','(.05,.1]','(.1,.25]','(.25,.5]','(.5,1]','(1,2]','(2,3]','>3']


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(values,positive):
    positive=np.asarray(positive,bool);n=int(positive.sum());m=len(positive)-n
    if not n or not m:return None
    _,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
    ranks=np.cumsum(counts)-(counts-1)/2
    return float((ranks[inverse][positive].sum()-n*(n+1)/2)/(n*m))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    complete=json.loads((args.evaluation/'complete.json').read_text())
    protocol=json.loads((args.evaluation/'summary.json').read_text())
    if protocol['split']!='test':raise ValueError('Only complete independent tests are accepted')
    paths=sorted(args.evaluation.glob('Safety_*.json'))+sorted(args.evaluation.glob('C1_*.json'))
    rows=[json.loads(f.read_text()) for f in paths]
    if len(rows)!=complete['episodes']:raise ValueError('Incomplete records')
    keys=[(r['rid'],r['noise_seed'],r['method']) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate records')
    if {(r,n) for r,n,m in keys if m=='Safety'}!={(r,n) for r,n,m in keys if m=='C1'}:
        raise ValueError('Unpaired records')
    for r in rows:
        if r['six_class_outcome'] not in LABELS:raise ValueError('Unexpected outcome')
        value=r['risk']['J_live']
        if not np.isfinite(value) or value<0:raise ValueError('Invalid risk')
    args.out.mkdir(parents=True)
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True,layout='constrained')
    bins=[];summary={}
    for ax,method in zip(axes,('Safety','C1')):
        group=[r for r in rows if r['method']==method]
        values=np.array([r['risk']['J_live'] for r in group])
        dead=np.array([r['six_class_outcome'] in LABELS[1:3] for r in group])
        indices=np.where(values==0,0,np.searchsorted(UPPER,values,side='left')+1)
        counts=np.zeros((len(BIN_NAMES),len(LABELS)),int)
        for r,i in zip(group,indices):counts[i,LABELS.index(r['six_class_outcome'])]+=1
        totals=counts.sum(axis=1);fractions=counts/np.maximum(totals[:,None],1)
        bottom=np.zeros(len(totals))
        for j,(label,color) in enumerate(zip(LABELS,COLORS)):
            ax.bar(np.arange(len(totals)),fractions[:,j],bottom=bottom,color=color,label=label)
            bottom+=fractions[:,j]
        for i,n in enumerate(totals):ax.text(i,1.025,str(n),ha='center',fontsize=8)
        ax.set_ylim(0,1.13);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('Fraction within risk bin')
        ax.set_title(f'{method}: n={len(group)}; numbers above bars are episode counts')
        ax.spines[['top','right']].set_visible(False)
        for i,name in enumerate(BIN_NAMES):
            bins.append(dict(method=method,risk_bin=name,n=int(totals[i]),**{label:int(counts[i,j]) for j,label in enumerate(LABELS)}))
        summary[method]=dict(n=len(group),retrospective_deadlock_auc=auc(values,dead),
            deadlocks_below_one=int(np.sum(dead&(values<1-1e-8))),
            non_deadlocks_at_least_one=int(np.sum(~dead&(values>=1))),
            zero_risk_ordinary_timeouts=sum(r['six_class_outcome']=='other_timeout' and r['risk']['J_live']==0 for r in group))
    axes[0].legend(loc='upper right',fontsize=8)
    axes[-1].set_xticks(range(len(BIN_NAMES)),BIN_NAMES,rotation=25,ha='right')
    axes[-1].set_xlabel('Full-trajectory risk (dimensionless, not probability)')
    fig.suptitle('Retrospective risk/outcome correspondence; not a 20-second forecast',fontsize=12)
    fig.savefig(args.out/'risk_correspondence.png',dpi=180)
    fig.savefig(args.out/'risk_correspondence.pdf')
    plt.close(fig)
    with (args.out/'risk_bins.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(bins[0]));writer.writeheader();writer.writerows(bins)
    (args.out/'summary.json').write_text(json.dumps(dict(summary=summary,
        interpretation='Retrospective association includes the event definition; not prospective prediction, calibrated probability or causal efficacy. Counts include correlated noise replicas; no inferential interval is claimed by this plot.',
        evaluation_protocol=protocol,script_sha256=digest(Path(__file__)),
        record_sha256={f.name:digest(f) for f in paths}),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
