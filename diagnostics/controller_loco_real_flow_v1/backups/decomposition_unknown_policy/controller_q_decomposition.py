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

def metrics(x):
    # x: 7 controllers x S shared physical states x 16 eta x 8 seeds.
    a=(x[:,:,:,0:4]==1).mean(-1);b=(x[:,:,:,4:8]==1).mean(-1)
    level=((a+b)/2).mean((1,2));eta=((a+b)/2).mean(1)
    ce_a=center(a.mean(1),(0,1));ce_b=center(b.mean(1),(0,1))
    triple_a=center(a,(0,1,2));triple_b=center(b,(0,1,2))
    return dict(level_sd=float(level.std()),controller_eta_energy_crossfit=float(np.mean(ce_a*ce_b)),
        state_controller_eta_energy_crossfit=float(np.mean(triple_a*triple_b)))

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
    base=metrics(x);rng=np.random.default_rng(2026101211);draw=rng.integers(64,size=(2000,64))
    boot=[metrics(x[:,ids]) for ids in draw]
    ci={key:np.quantile([v[key] for v in boot],[.025,.975]).tolist() for key in base}
    unknown=(x==-1).mean((1,2,3))
    result=dict(status='AUXILIARY DESCRIPTIVE',controllers=names,states=64,eta=16,seeds=8,
        unknown_policy='crossfit energy treats unknown as zero success; unknown rates separately retained',
        unknown_rate_by_controller=dict(zip(names,unknown.tolist())),
        controller_mean_q_observed=dict(zip(names,np.nanmean(q,axis=(1,2,3)).tolist())),
        best_fixed_eta_observed=dict(zip(names,mean.argmax(-1).astype(int).tolist())),
        metrics=base,ci95_state_bootstrap=ci,bootstrap_seed=2026101211,bootstrap_draws=2000,
        pairwise_eta_rankings=rankings,
        caveat='K16/Q8 and two seed halves; interaction energy is an auxiliary lower-label estimate, not continuous success-basin identification')
    write(HERE/'CONTROLLER_Q_DECOMPOSITION.json',result)
    print(json.dumps(dict(metrics=base,ci=ci,best=result['best_fixed_eta_observed']),indent=2))

if __name__=='__main__':main()
