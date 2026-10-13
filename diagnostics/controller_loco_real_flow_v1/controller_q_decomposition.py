"""Auxiliary held-panel Q level versus η-ordering decomposition."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def center(x,axes):
    for ax in axes:x=x-x.mean(axis=ax,keepdims=True)
    return x

def measure(x):
    # x: 7 controllers x S shared physical states x 16 eta x 8 seeds.
    # Unknowns remain intervals [0,1], never binary failures. Midpoints are
    # numerical summaries only; contraction/Cauchy bounds cover every unknown
    # completion, since the balanced centering maps are orthogonal projections.
    a=((x[:,:,:,0:4]==1)+.5*(x[:,:,:,0:4]==-1)).mean(-1)
    b=((x[:,:,:,4:8]==1)+.5*(x[:,:,:,4:8]==-1)).mean(-1)
    ua=(x[:,:,:,0:4]==-1).mean(-1);ub=(x[:,:,:,4:8]==-1).mean(-1)
    level=((a+b)/2).mean((1,2));eta=((a+b)/2).mean(1)
    ce_a=center(a.mean(1),(0,1));ce_b=center(b.mean(1),(0,1))
    triple_a=center(a,(0,1,2));triple_b=center(b,(0,1,2))
    values=dict(level_sd=float(level.std()),controller_eta_energy_crossfit=float(np.mean(ce_a*ce_b)),
        state_controller_eta_energy_crossfit=float(np.mean(triple_a*triple_b)))
    norm=lambda v:float(np.sqrt(np.mean(v*v)))
    radius=.25*norm((ua+ub).mean((1,2)))
    bounds={'level_sd':[max(0,values['level_sd']-radius),values['level_sd']+radius]}
    for key,pa,pb,ra,rb in [
        ('controller_eta_energy_crossfit',ce_a,ce_b,.5*norm(ua.mean(1)),.5*norm(ub.mean(1))),
        ('state_controller_eta_energy_crossfit',triple_a,triple_b,.5*norm(ua),.5*norm(ub))]:
        error=ra*norm(pb)+rb*norm(pa)+ra*rb
        bounds[key]=[values[key]-error,values[key]+error]
    return values,bounds

def metrics(x):return measure(x)[0]

def main():
    p=read(HERE/'protocol.json');names=p['controller_order']
    x=np.load(HERE/'LABELS.npz')['outcomes'][:,128:,:,:]
    assert x.shape==(7,64,16,8) and np.all(x>=-1)
    q=np.where(x==1,1.,np.where(x==0,0.,np.nan));mean=np.nanmean(q,axis=(1,3))
    rankings=[]
    for a in range(7):
        for b in range(a+1,7):
            rho=spearmanr(mean[a],mean[b]).statistic
            rankings.append(dict(a=names[a],b=names[b],rho=float(rho) if np.isfinite(rho) else None,
                differing_best_eta=bool(int(mean[a].argmax())!=int(mean[b].argmax()))))
    base,bounds=measure(x);rng=np.random.default_rng(2026101211);draw=rng.integers(64,size=(2000,64))
    boot=[measure(x[:,ids]) for ids in draw]
    ci={key:[float(np.quantile([v[1][key][0] for v in boot],.025)),float(np.quantile([v[1][key][1] for v in boot],.975))] for key in base}
    midpoint_ci={key:np.quantile([v[0][key] for v in boot],[.025,.975]).tolist() for key in base}
    unknown=(x==-1).mean((1,2,3))
    result=dict(status='AUXILIARY DESCRIPTIVE',controllers=names,states=64,eta=16,seeds=8,
        unknown_policy='unknowns retained as [0,1]; midpoint summaries plus conservative orthogonal-projection/Cauchy completion bounds; no unknown is recoded as failure',
        unknown_rate_by_controller=dict(zip(names,unknown.tolist())),
        controller_mean_q_observed=dict(zip(names,np.nanmean(q,axis=(1,2,3)).tolist())),
        best_fixed_eta_observed=dict(zip(names,mean.argmax(-1).astype(int).tolist())),
        metrics=base,unknown_completion_bounds=bounds,ci95_state_bootstrap=ci,
        ci95_midpoint_state_bootstrap=midpoint_ci,bootstrap_seed=2026101211,bootstrap_draws=2000,
        pairwise_eta_rankings=rankings,
        caveat='K16/Q8; auxiliary interaction estimates. Noise-debiasing is exact only for complete labels with independent halves; arbitrary unknownness prevents an unconditional unbiasedness claim. CIs include unknown-completion sensitivity; no continuous basin identification.')
    write(HERE/'CONTROLLER_Q_DECOMPOSITION.json',result)
    print(json.dumps(dict(metrics=base,ci=ci,best=result['best_fixed_eta_observed']),indent=2))

if __name__=='__main__':main()
