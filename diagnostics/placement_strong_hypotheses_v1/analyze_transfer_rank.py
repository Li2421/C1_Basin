"""Test whether a single initial-state G contrast ranks finite boundary shifts."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'


def main():
    roots=json.loads((HERE/'h1_roots.json').read_text())
    lines=json.loads((HERE/'h1_line_protocol.json').read_text())['lines']
    z=np.load(PRIOR/'raw_G_M.npz');rows=[]
    for line in lines:
        if not line['available']:continue
        uid=line['uid'];a=next(x for x in roots if x['uid']==uid and x['chain']=='TT');b=next(x for x in roots if x['uid']==uid and x['chain']=='FF')
        d=np.asarray(line['endpoint_eta'][1])-line['endpoint_eta'][0]
        gt=z[uid+'_G_TT'];gf=z[uid+'_G_FF']
        rows.append(dict(uid=uid,unique_both=a['crossing_count_on_original_grid']==b['crossing_count_on_original_grid']==1,
            true_signed_fraction_shift=b['first_root_fraction']-a['first_root_fraction'],
            true_eta_distance=abs(b['first_root_fraction']-a['first_root_fraction'])*np.linalg.norm(d),
            initial_G_fro_difference=float(np.linalg.norm(gf-gt)),
            initial_direction_transfer_difference=float(np.linalg.norm((gf-gt)@d)),
            initial_TT_direction_norm=float(np.linalg.norm(gt@d)),
            initial_FF_direction_norm=float(np.linalg.norm(gf@d))))
    corr={}
    for subset_name,ss in (('all_first_crossing_descriptive',rows),('unique_crossings_only',[r for r in rows if r['unique_both']])):
        x=[r['initial_direction_transfer_difference'] for r in ss];y=[r['true_eta_distance'] for r in ss]
        corr[subset_name]=dict(n=len(ss),spearman_rho=float(spearmanr(x,y).statistic) if len(ss)>2 else None,
            warning='first crossing is nonunique as a boundary correspondence' if subset_name.startswith('all') else 'only three states')
    out=dict(rows=rows,rank_correlations=corr,
             note='Initial G alone is neither a predicted boundary shift nor the path-integrated state-dependent transfer.')
    (HERE/'transfer_boundary_rank.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(corr,indent=2))


if __name__=='__main__':main()
