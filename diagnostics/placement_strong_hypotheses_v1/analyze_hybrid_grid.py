"""Test whether finite action-family swaps track donor or baseline on five lines."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent


def roots(y):
    out=[]
    for i in range(len(y)-1):
        if (y[i]>0)!=(y[i+1]>0):
            out.append((i-y[i]/(y[i+1]-y[i]))/16)
    return out


def main():
    p=json.loads((HERE/'hybrid_grid_protocol.json').read_text())
    endpoints=json.loads((HERE/'h1_endpoint_replay.json').read_text())
    summaries=[];all_rows=[]
    fig,axes=plt.subplots(len(p['lines']),1,figsize=(9,2.5*len(p['lines'])),sharex=True)
    for ax,line in zip(axes,p['lines']):
        uid=line['uid'];normal={}
        for ch in ('TT','FF'):
            y=[]
            for i in range(17):
                if i in (0,16):
                    r=next(x for x in endpoints if x['uid']==uid and x['chain']==ch and x['fraction']==i/16)
                else:r=json.loads((HERE/'h1_line_results'/f'{uid}_{ch}_t{i:02d}.json').read_text())
                y.append(r)
            normal[ch]=y
            ax.plot(np.arange(17)/16,[r['rho'] for r in y],label=ch,lw=1)
        for mode,color in (('FF_base_TT_corr','#2ca02c'),('TT_base_FF_corr','#9467bd')):
            donor='TT' if mode.startswith('FF') else 'FF';base='FF' if donor=='TT' else 'TT'
            hybrids=[json.loads((HERE/'hybrid_grid_results'/f'{uid}_{mode}_{i:02d}.json').read_text()) for i in range(17)]
            donor_labels=[r['rho']>0 for r in normal[donor]]
            base_labels=[r['rho']>0 for r in normal[base]]
            h_labels=[r['rho']>0 for r in hybrids]
            hm=np.asarray([r['rho'] for r in hybrids]);dm=np.asarray([r['rho'] for r in normal[donor]]);bm=np.asarray([r['rho'] for r in normal[base]])
            ratio=[r['closure_mean']/max(r['donor_correction_mean'],1e-12) for r in hybrids]
            summary=dict(uid=uid,mode=mode,donor=donor,base=base,points=17,
                match_donor=sum(a==b for a,b in zip(h_labels,donor_labels)),
                match_base=sum(a==b for a,b in zip(h_labels,base_labels)),
                donor_baseline_label_disagreements=sum(a!=b for a,b in zip(donor_labels,base_labels)),
                rho_mae_to_donor=float(np.mean(np.abs(hm-dm))),rho_mae_to_base=float(np.mean(np.abs(hm-bm))),
                normal_donor_roots=roots(dm),hybrid_roots=roots(hm),normal_base_roots=roots(bm),
                median_closure_to_donor=float(np.median(ratio)),max_instant_closure=float(max(r['closure_max'] for r in hybrids)),
                rho_success_sign_mismatches=sum((r['rho']>0)!=r['success'] for r in hybrids))
            summaries.append(summary)
            for h,donor_label,base_label in zip(hybrids,donor_labels,base_labels):
                all_rows.append(dict(uid=uid,mode=mode,fraction=h['fraction'],
                    donor_label=donor_label,base_label=base_label,hybrid_label=bool(h['success']),
                    donor_match=bool(h['success']==donor_label),base_match=bool(h['success']==base_label),
                    closure_relative_mean=h['closure_mean']/max(h['donor_correction_mean'],1e-12),
                    closure_max=h['closure_max']))
            ax.plot(np.arange(17)/16,hm,label=mode,color=color,alpha=.8,lw=1)
        ax.axhline(0,color='black',lw=.6);ax.set_ylabel(uid[-6:]+' rho');ax.legend(fontsize=6,ncol=4)
    axes[-1].set_xlabel('candidate-line fraction');fig.tight_layout();(HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/hybrid_grid_lines.pdf');fig.savefig(HERE/'figures/hybrid_grid_lines.png',dpi=180);plt.close(fig)
    import statistics
    agreement={}
    for label in (True,False):
        rr=[x for x in all_rows if x['donor_match']==label]
        agreement[str(label)]=dict(count=len(rr),median_closure_relative_mean=statistics.median(x['closure_relative_mean'] for x in rr),
            median_closure_max=statistics.median(x['closure_max'] for x in rr))
    (HERE/'hybrid_grid_analysis.json').write_text(json.dumps(dict(summary=summaries,records=all_rows,
        aggregate=dict(total=len(all_rows),donor_matches=sum(x['donor_match'] for x in all_rows),
            baseline_matches=sum(x['base_match'] for x in all_rows),
            donor_baseline_disagreements=sum(x['donor_label']!=x['base_label'] for x in all_rows),
            on_disagreement_donor_matches=sum(x['donor_match'] for x in all_rows if x['donor_label']!=x['base_label']),
            closure_by_donor_match=agreement)),indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
