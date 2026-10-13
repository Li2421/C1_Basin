"""Aggregate regular Ring local root predictions, including prior two states."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'


def main():
    new=json.loads((HERE/'h1_gradient_analysis.json').read_text())['boundary_predictions']
    old=json.loads((PRIOR/'cross_boundary_shift_predictions.json').read_text())
    same_eta=json.loads((PRIOR/'same_eta_boundary_chain.json').read_text())
    lines={x['uid']:x for x in json.loads((HERE/'h1_line_protocol.json').read_text())['lines']}
    prior_lines={x['uid']:x for x in json.loads((PRIOR/'boundary_protocol.json').read_text())}
    rows=[]
    for x in new:
        rows.append(dict(uid=x['uid'],source=x['source_root'],target=x['target'],
                         true_shift=x['true_signed_fraction_displacement'],
                         predicted_shift=x['predicted_signed_fraction_displacement'],
                         signed_target_normal_displacement=x['signed_target_normal_displacement'],
                         line_norm=float(np.linalg.norm(np.asarray(lines[x['uid']]['endpoint_eta'][1])-lines[x['uid']]['endpoint_eta'][0])),
                         gradient_stability=x['gradient_stability'],branch_stable=x['branch_stable'],origin='new'))
    for x in old:
        uid=x['state_uid'];source=x['source_boundary_chain'];target=x['chain']
        d=np.asarray(prior_lines[uid]['endpoint_eta'][1])-prior_lines[uid]['endpoint_eta'][0]
        matched=next(r for r in same_eta if r['uid']==uid and r['eta_source_boundary']==source)
        grad=np.asarray(matched[target]['grad_rho'])
        true=x['own_boundary_t']-x['source_boundary_t']
        rows.append(dict(uid=uid,source=source,target=target,
            true_shift=true,
            predicted_shift=x['predicted_root_from_other_point']-x['source_boundary_t'],
            signed_target_normal_displacement=float(true*(grad@d)/max(np.linalg.norm(grad),1e-12)),
            line_norm=float(np.linalg.norm(np.asarray(prior_lines[uid]['endpoint_eta'][1])-prior_lines[uid]['endpoint_eta'][0])),
            origin='prior'))
    for x in rows:
        x['true_eta_distance']=abs(x['true_shift'])*x['line_norm']
        x['predicted_eta_distance']=abs(x['predicted_shift'])*x['line_norm']
        x['eta_magnitude_error']=abs(abs(x['predicted_shift'])-abs(x['true_shift']))*x['line_norm']
        x['eta_vector_error']=abs(x['predicted_shift']-x['true_shift'])*x['line_norm']
        x['displacement_cosine_on_fixed_line']=float(np.sign(x['predicted_shift']*x['true_shift']))
        x['absolute_fraction_error']=abs(x['predicted_shift']-x['true_shift'])
    main=[x for x in rows if x['source']=='TT' and x['target']=='FF']
    corr=dict(n=len(main),spearman_signed=float(spearmanr([x['predicted_shift'] for x in main],[x['true_shift'] for x in main]).statistic),
              spearman_magnitude=float(spearmanr([x['predicted_eta_distance'] for x in main],[x['true_eta_distance'] for x in main]).statistic),
              median_eta_vector_error=float(np.median([x['eta_vector_error'] for x in main])),
              maximum_eta_vector_error=float(max(x['eta_vector_error'] for x in main)))
    out=dict(pairs=rows,TT_to_FF_rank_correlation=corr,
        caution='The cosine is only a sign on a prespecified one-dimensional line. Predictions use full measured margin/path derivative, not G alone; four regular states are too few for a population claim.')
    (HERE/'h1_boundary_prediction.json').write_text(json.dumps(out,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(5,5))
    ax.scatter([x['true_shift'] for x in main],[x['predicted_shift'] for x in main])
    for x in main:ax.annotate(x['uid'][-6:],(x['true_shift'],x['predicted_shift']))
    rng=[min(min(x['true_shift'],x['predicted_shift']) for x in main),max(max(x['true_shift'],x['predicted_shift']) for x in main)]
    ax.plot(rng,rng,'k--',lw=.8);ax.set_xlabel('measured TT→FF line displacement');ax.set_ylabel('local margin-chain prediction')
    fig.tight_layout();(HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/boundary_prediction.pdf');fig.savefig(HERE/'figures/boundary_prediction.png',dpi=180);plt.close(fig)
    print(json.dumps(corr,indent=2))


if __name__=='__main__':main()
