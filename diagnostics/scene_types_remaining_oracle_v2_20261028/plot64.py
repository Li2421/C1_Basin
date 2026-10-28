"""Standalone scientific budget curves; read completed evidence only."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'plot_cache')
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    assert json.loads((HERE/'ORACLE64_COMPLETE.json').read_text())['complete']
    results=json.loads((HERE/'ORACLE64_RESULTS.json').read_text())
    with (HERE/'oracle64_budget_curve.csv').open()as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for s in results:
        rr=[r for r in rows if r['family']==s['family']]
        label='+'.join(s['columns'])+(' (G>=0)' if s['family']=='B3_wide'else'')
        axes[0].step([int(r['K'])for r in rr],[int(r['covered'])for r in rr],where='post',label=label)
    axes[0].set(xlabel='Candidate budget K',ylabel='Covered cases /120',xlim=(1,64),title='Finite correction-pool coverage')
    axes[0].axvline(16,color='grey',linestyle=':',linewidth=1)
    axes[0].axvline(32,color='grey',linestyle=':',linewidth=1)
    axes[0].legend(fontsize=7,loc='lower right');axes[0].grid(alpha=.2)
    names=['+'.join(s['columns'])+('\nG>=0' if s['family']=='B3_wide'else'')for s in results]
    x=list(range(len(results)))
    axes[1].bar([i-.18for i in x],[s['first_success_index_mean']for s in results],width=.36,label='Successful cases only')
    axes[1].bar([i+.18for i in x],[s['all_cohort_restricted_query_mean']for s in results],width=.36,label='All cases; failure=64')
    axes[1].set_xticks(x,names,rotation=45,ha='right',fontsize=7)
    axes[1].set(ylabel='Mean candidate queries',title='Search effort in the frozen pool order',ylim=(0,64));axes[1].legend(fontsize=8)
    fig.savefig(HERE/'oracle64_budget_and_queries.svg')
    fig.savefig(HERE/'oracle64_budget_and_queries.png',dpi=160)
    plt.close(fig)

if __name__=='__main__':main()
