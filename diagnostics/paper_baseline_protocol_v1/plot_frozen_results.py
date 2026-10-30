"""Standalone scientific figures from saved JSON only; no simulation imports."""
import guard
from guard import HERE
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from method_names import NAMES

LABELS=NAMES
plt.rcParams['font.family']=['Noto Serif CJK SC','DejaVu Serif']
plt.rcParams['axes.unicode_minus']=False


def run():
    folder=HERE/'figures';folder.mkdir(exist_ok=True);saved=[]
    tuning=json.loads((HERE/'TUNING_CURVES.json').read_text())['results']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for key,value in tuning.items():
        benchmark,method=key.split('::');curve=value.get('full_curve')
        if not curve:continue
        axes[1 if benchmark=='B'else 0].plot([r['k']for r in curve],[r['expected_max']for r in curve],label=LABELS.get(method,method))
    for ax,title in zip(axes,['Benchmark A training','Benchmark B training']):
        ax.set(title=title,xlabel='Number of distinct settings',ylabel='Expected best scene-equal success',ylim=(0,1));ax.grid(alpha=.2);ax.legend()
    path=folder/'training_budget_curves.pdf';fig.savefig(path);saved.append(str(path));fig.savefig(folder/'training_budget_curves.png',dpi=180);plt.close(fig)
    oracle=HERE/'ORACLE16_RESULTS.json'
    if oracle.exists():
        fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
        for method in json.loads(oracle.read_text())['methods']:
            curve=[r for r in method.get('budget_curve',[])if r['complete']]
            if not curve:continue
            cases=method['summary']['cases']
            ax.step([r['k']for r in curve],[r['verified_coverage']/cases for r in curve],where='post',label=LABELS.get(method['method'],method['method']))
        ax.set(xlabel='Frozen prefix budget K',ylabel='Verified nominal-case coverage',ylim=(0,1),xlim=(1,16));ax.grid(alpha=.2);ax.legend()
        path=folder/'oracle_prefix_coverage.pdf';fig.savefig(path);saved.append(str(path));fig.savefig(folder/'oracle_prefix_coverage.png',dpi=180);plt.close(fig)
    (folder/'MANIFEST.json').write_text(json.dumps(dict(source='Saved frozen JSON only; no rollout',files=saved,
        interpretation='Training curves use exact random-subset expected maxima. Oracle figure shows complete frozen prefixes; nominal cases shown, physical-block intervals are in JSON.'),indent=2)+'\n')
    return saved


if __name__=='__main__':print(run())
