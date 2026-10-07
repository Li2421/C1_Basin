"""Paired source-family uncertainty; repeated controllers/seeds are clustered."""
import csv
import numpy as np
from .experiment import OUT,SEEDS,KINDS,read,write,csvwrite

def main():
    d=np.load(OUT/'source_data.npz');rng=np.random.default_rng(20261004123)
    family_losses={};picks={}
    for kind in KINDS:
      for seed in SEEDS:
        losses=np.zeros(46);b15=np.zeros((46,2),bool)
        for fold in range(3):
            p=OUT/'models'/kind/f'fold{fold}_seed{seed}'
            a=np.load(p/'predictions.npz');ii=a['indices'];z=a['best_correct']
            ss=d['success'][:,ii];ff=d['failure'][:,ii]
            terms=ss*np.logaddexp(0,-z)+ff*np.logaddexp(0,z)
            for s in np.unique(d['state_index'][ii]):
                pos=np.flatnonzero(d['state_index'][ii]==s)
                losses[s]=terms[:,pos].sum()/(ss[:,pos]+ff[:,pos]).sum()
                for c in range(2):b15[s,c]=ss[c,pos[np.argmax(z[c,pos])]]>=15
        family_losses[kind,seed]=losses;picks[kind,seed]=b15
    res=[]
    for kind in KINDS[1:]:
      ref=('wide_db_eta_only' if kind.startswith('wide_db_full') else 'wide_eta_only' if kind=='wide_full_scaled' else 'eta_only')
      for seed in SEEDS:
        delta=family_losses[ref,seed]-family_losses[kind,seed]
        draw=rng.integers(0,46,(20000,46));ci=np.quantile(delta[draw].mean(1),[.025,.975])
        a=picks[kind,seed];b=picks[ref,seed];effect=a.astype(int)-b.astype(int)
        bci=np.quantile(effect.sum(1)[draw].mean(1)/2,[.025,.975])
        res.append(dict(kind=kind,reference=ref,seed=seed,NLL_gain=float(delta.mean()),
            NLL_gain_95lo=float(ci[0]),NLL_gain_95hi=float(ci[1]),
            rescue=int((a&~b).sum()),break_count=int((~a&b).sum()),
            selection_gain_95lo=float(bci[0]),selection_gain_95hi=float(bci[1])))
    csvwrite(OUT/'paired_uncertainty.csv',res)
    trajectories=[]
    for kind in KINDS:
      for fold in range(3):
       for seed in SEEDS:
        p=OUT/'models'/kind/f'fold{fold}_seed{seed}'
        meta=read(p/'complete.json');hist=read(p/'history.json');best=next(r for r in hist if r['step']==meta['best_step'])
        trajectories.append(dict(kind=kind,fold=fold,seed=seed,best_step=meta['best_step'],
            fit_NLL_best=best['train']['NLL'],inner_NLL_best=best['inner']['NLL'],
            fit_NLL_last=hist[-1]['train']['NLL'],inner_NLL_last=hist[-1]['inner']['NLL']))
    csvwrite(OUT/'fit_gap_audit.csv',trajectories)
    print([r for r in res if r['kind']=='full_raw'])

if __name__=='__main__':main()
