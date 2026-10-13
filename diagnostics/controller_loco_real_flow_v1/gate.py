"""Discovery-seed-only controller necessity and cross-controller regret."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def best(q):
    # Unknown is excluded from η selection. Ties resolve to the lowest K16 index.
    suc=(q==1).sum(axis=(-3,-1));valid=((q==1)|(q==0)).sum(axis=(-3,-1))
    return np.divide(suc,valid,out=np.full_like(suc,-1,dtype=float),where=valid>0).argmax(-1)

def main():
    p=read(HERE/'protocol.json');names=p['controller_order'];out=np.load(HERE/'LABELS.npz')['outcomes']
    train=out[:,:128,:,:4];test=out[:,128:,:,:4]
    source_curve=[]
    for c in range(7):
        suc=(train[c]==1).sum(-1);valid=((train[c]==1)|(train[c]==0)).sum(-1)
        q=np.divide(suc,valid,out=np.full_like(suc,np.nan,dtype=float),where=valid>0)
        source_curve.append(np.nanmean(q,axis=0).tolist())
    source_curve=np.asarray(source_curve)
    fixed_train=source_curve.argmax(-1)
    rng=np.random.default_rng(2026101205);draws=rng.integers(64,size=(2000,64))
    rows=[];pairs=[]
    for held in range(7):
        sources=[c for c in range(7) if c!=held]
        src=int(source_curve[sources].mean(0).argmax())
        x=test[held]
        # Two mutually independent seed halves for selecting target best fixed η.
        halves=[(slice(0,2),slice(2,4)),(slice(2,4),slice(0,2))]
        choices=[];lo=[];hi=[];target=[];srcvalue=[];srcupper=[];stateoracle=[]
        for discover,evaluate in halves:
            q=x[:,:,discover];suc=(q==1).sum((0,2));valid=((q==1)|(q==0)).sum((0,2))
            rate=np.divide(suc,valid,out=np.full(16,-1.),where=valid>0)
            chosen=int(rate.argmax());choices.append(chosen)
            a=x[:,chosen,evaluate];b=x[:,src,evaluate]
            av=(a==1).mean(-1);bv=(b==1).mean(-1)
            bu=(b==-1).mean(-1)*(chosen!=src)
            lo.append(av-bv-bu)
            hi.append(av+(a==-1).mean(-1)*(chosen!=src)-bv)
            target.append(av);srcvalue.append(bv);srcupper.append(bu)
            # Per-state cross-fit oracle is descriptive; Q2 selection is noisy.
            cs=np.where(q==1,1.,np.where(q==0,0.,np.nan))
            validstate=np.isfinite(cs).sum(-1);meanstate=np.divide(np.nansum(cs,-1),validstate,
                out=np.full((64,16),-1.),where=validstate>0)
            choice=meanstate.argmax(-1)
            stateoracle.append((x[np.arange(64),choice,evaluate]==1).mean(-1))
        lower=np.mean(lo,axis=0);upper=np.mean(hi,axis=0)
        ci=np.quantile(lower[draws].mean(1),[.025,.975])
        necessary=bool(lower.mean()>=.05 and ci[0]>0)
        row=dict(controller=names[held],fold=held,source_prior_eta=src,target_fixed_eta_crossfit=choices,
                 target_train_best_eta=int(fixed_train[held]),target_crossfit_fixed_success_lower=float(np.mean(target)),
                 source_prior_success_lower=float(np.mean(srcvalue)),source_prior_unknown_upper=float(np.mean(srcupper)),
                 necessity_loss_lower=float(lower.mean()),necessity_loss_upper=float(upper.mean()),
                 necessity_ci95_conservative=ci.tolist(),necessary=necessary,
                 state_oracle_crossfit_success_lower=float(np.mean(stateoracle)),
                 state_oracle_gain_over_source_lower=float(np.mean(stateoracle)-np.mean(srcvalue)))
        rows.append(row)
        for source in sources:
            own=int(fixed_train[source]);a=x[:,own,:];b=x[:,int(fixed_train[held]),:]
            regret=float((b==1).mean()-(a==1).mean())
            pairs.append(dict(source=names[source],target=names[held],source_train_eta=own,target_train_eta=int(fixed_train[held]),
                              cross_controller_regret_lower=regret))
    necessary=[r['fold'] for r in rows if r['necessary']]
    result=dict(status='COMPLETED DISCOVERY ONLY',states=64,seed_indices=[0,1,2,3],
                source_prior_method='equal-controller mean of per-state observed Q over selected train η',
                unknown_policy='unknown excluded from selection; paired conservative lower for gate; same-η unknown cancels',
                bootstrap_draws=2000,bootstrap_seed=2026101205,rows=rows,cross_controller_regret=pairs,
                necessary_folds=necessary,necessary_count=len(necessary),primary_test_eligible=len(necessary)>=2)
    write(HERE/'GATE_RESULTS.json',result)
    print(json.dumps(dict(necessary_folds=necessary,rows=[dict(controller=r['controller'],eta_src=r['source_prior_eta'],
        target_eta=r['target_fixed_eta_crossfit'],L=r['necessity_loss_lower'],ci=r['necessity_ci95_conservative']) for r in rows]),indent=2))

if __name__=='__main__':main()
