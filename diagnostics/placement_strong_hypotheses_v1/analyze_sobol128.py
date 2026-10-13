"""Independent QMC refinement and bounded-domain volume sensitivity."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent


def main():
    original=json.loads((HERE/'panel_analysis.json').read_text())
    p=json.loads((HERE/'sobol128_protocol.json').read_text())
    rows=[]
    for st in p['states']:
        uid=st['uid'];pair=[];num=0;lower=upper=0
        for i in range(128):
            a=json.loads((HERE/'sobol128_results'/f'{uid}_sobol128b_{i:03d}_TT.json').read_text())
            b=json.loads((HERE/'sobol128_results'/f'{uid}_sobol128b_{i:03d}_FF.json').read_text())
            ua=a['termination']=='numerical_failure';ub=b['termination']=='numerical_failure'
            num+=ua+ub
            lower+=int(bool(b['success']))-int(bool(a['success']) or ua)
            upper+=int(bool(b['success']) or ub)-int(bool(a['success']))
            pair.append((bool(a['success']),bool(b['success'])))
        old=next(x for x in original['states'] if x['uid']==uid)
        gain=sum(b and not a for a,b in pair)/128;loss=sum(a and not b for a,b in pair)/128
        delta=gain-loss
        pooled=(64*old['delta_fraction']+128*delta)/192
        old_lo,old_hi=old['delta_fraction_solver_uncertainty_interval']
        rows.append(dict(uid=uid,scene=st['scenario'],sobol64_delta=old['delta_fraction'],
            independent_sobol128_delta=delta,pooled192_delta=pooled,
            independent_solver_uncertainty_interval=[lower/128,upper/128],
            pooled_solver_uncertainty_interval=[(64*old_lo+lower)/192,(64*old_hi+upper)/192],
            independent_gain=gain,independent_loss=loss,
            absolute_quadrature_discrepancy=abs(delta-old['delta_fraction']),
            sign_stable_across_scrambles=bool(np.sign(delta)==np.sign(old['delta_fraction'])),
            numerical_failures=int(num)))
    rng=np.random.default_rng(20261009)
    stats={}
    for scene in ('ring_exchange','toy_give_way','both'):
        rr=[r for r in rows if scene=='both' or r['scene']==scene]
        v=np.asarray([r['pooled192_delta'] for r in rr])
        boot=np.mean(v[rng.integers(0,len(v),size=(20000,len(v)))],axis=1)
        stats[scene]=dict(states=len(rr),mean_64=float(np.mean([r['sobol64_delta'] for r in rr])),
            mean_independent_128=float(np.mean([r['independent_sobol128_delta'] for r in rr])),
            mean_pooled_192=float(np.mean(v)),median_pooled_192=float(np.median(v)),
            mean_solver_uncertainty_interval=[float(np.mean([r['pooled_solver_uncertainty_interval'][j] for r in rr])) for j in (0,1)],
            paired_state_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
            states_with_both_scrambles_positive=sum(r['sobol64_delta']>0 and r['independent_sobol128_delta']>0 for r in rr),
            states_with_both_scrambles_negative=sum(r['sobol64_delta']<0 and r['independent_sobol128_delta']<0 for r in rr),
            median_absolute_quadrature_discrepancy=float(np.median([r['absolute_quadrature_discrepancy'] for r in rr])))
    out=dict(rows=rows,stats=stats,box=original['box'],box_volume=original['box_volume'],
             note='Outcome-blind all-state follow-up after coarse numerical discrepancy; 64 and 128 are separate scrambled QMC designs, not confidence replicates of equal size.')
    (HERE/'sobol128_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6,5))
    for scene,color in (('ring_exchange','#1f77b4'),('toy_give_way','#ff7f0e')):
        rr=[r for r in rows if r['scene']==scene]
        ax.scatter([r['sobol64_delta'] for r in rr],[r['independent_sobol128_delta'] for r in rr],color=color,label=scene)
        for r in rr:ax.annotate(r['uid'][-6:],(r['sobol64_delta'],r['independent_sobol128_delta']),fontsize=7)
    ax.axhline(0,color='black',lw=.6);ax.axvline(0,color='black',lw=.6);ax.plot([-.5,.5],[-.5,.5],'k--',lw=.6)
    ax.set_xlabel('Sobol64 volume fraction difference');ax.set_ylabel('independent Sobol128 difference');ax.legend()
    fig.tight_layout();(HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/volume_qmc_refinement.pdf');fig.savefig(HERE/'figures/volume_qmc_refinement.png',dpi=180);plt.close(fig)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
