"""Model-free selectable space, with unsupported cross-fitting left explicit."""
from common import *

BASE=12


def best(scores):
    scores=np.asarray(scores);choice=np.argmax(scores,axis=-1)
    return np.where(scores[...,BASE]>=scores.max(-1)-1e-12,BASE,choice)


def average(values,types,type_equal=True):
    per=values.mean(tuple(range(1,values.ndim)))
    return float(np.mean([per[types==t].mean() for t in sorted(set(types))])) if type_equal else float(per.mean())


def sample(types,ns,rng):
    groups=sorted(set(types));indices=[];weights=[]
    for t in rng.choice(groups,len(groups),replace=True):
        pool=np.flatnonzero(types==t);draw=rng.choice(pool,len(pool),replace=True)
        indices.extend(draw);weights.extend([1/(len(groups)*len(pool))]*len(pool))
    cc=np.asarray(indices);ss=rng.integers(ns,size=(len(cc),ns))
    return cc,ss,np.asarray(weights)


def statistic(lower,upper,mode):
    # Seed0 is the common evidence shared by all thirteen training actions.
    lo=lower[...,0];hi=upper[...,0];n,s=lo.shape[:2]
    if mode=='state':choice=best(lo)
    elif mode=='controller':choice=np.broadcast_to(best(lo.mean(1))[:,None],(n,s))
    else:raise ValueError(mode)
    chosen_lo=lo[np.arange(n)[:,None],np.arange(s)[None],choice]
    chosen_hi=hi[np.arange(n)[:,None],np.arange(s)[None],choice]
    same=choice==BASE
    return np.stack((chosen_lo-lo[...,BASE],np.where(same,0.,chosen_lo-hi[...,BASE]),np.where(same,0.,chosen_hi-lo[...,BASE])),axis=-1).mean(1)


def comparison(labels,types,mode,type_equal):
    lower=(labels==1).astype(float);upper=(labels!=0).astype(float)
    values=statistic(lower,upper,mode)
    point=np.mean([values[types==t].mean(0) for t in sorted(set(types))],axis=0) if type_equal else values.mean(0)
    rng=np.random.default_rng(202610180901);draws=[];n,ns=labels.shape[:2]
    for _ in range(5000):
        if type_equal:cc,ss,w=sample(types,ns,rng)
        else:
            cc=rng.integers(n,size=n);ss=rng.integers(ns,size=(n,ns));w=np.full(n,1/n)
        a=lower[cc[:,None],ss];b=upper[cc[:,None],ss]
        draws.append(np.sum(statistic(a,b,mode)*w[:,None],axis=0))
    draws=np.asarray(draws)
    return dict(gain_pp=float(point[0]*100),unknown_bounds_pp=(point[1:]*100).tolist(),
        CI95_pp=[float(np.quantile(draws[:,1],.025)*100),float(np.quantile(draws[:,2],.975)*100)],
        evidence='IN_SAMPLE_ON_COMMON_SEED0',replicates=5000,oracle_reselected_in_bootstrap=True)


def main(which):
    assert which in ('pilot','all')
    name='PILOT_TRAINING_BUNDLE' if which=='pilot' else 'TRAINING_BUNDLE'
    receipt=HERE/f'{name}.receipt.json'
    for source,digest in read(receipt)['files'].items():assert sha(source)==digest
    with np.load(HERE/f'{name}.npz',allow_pickle=False) as z:labels=z['labels'];types=z['types'];names=z['names']
    assert not np.any(labels[...,0]==-2)
    lower=(labels[...,0]==1).astype(float);upper=(labels[...,0]!=0).astype(float)
    generator_lo=lower[:,:,:10].max(-1);generator_hi=upper[:,:,:10].max(-1)
    lo=lower[:,:,BASE];hi=upper[:,:,BASE]
    def describe(mask,te):
        tt=types[mask];l=labels[mask]
        return dict(controllers=int(mask.sum()),types=sorted(set(tt.tolist())),
            progress_schedule_success_lower_pct=100*average(lo[mask],tt,te),
            state_in_sample_oracle_vs_progress=comparison(l,tt,'state',te),
            controller_in_sample_oracle_vs_progress=comparison(l,tt,'controller',te),
            generator_coverage=dict(empirical_nonworse_fraction=average((generator_lo>=lo)[mask],tt,te),
                empirical_strict_better_fraction=average((generator_lo>lo)[mask],tt,te),
                nonworse_unknown_bounds=[average((generator_lo>=hi)[mask],tt,te),average((generator_hi>=lo)[mask],tt,te)],
                strict_better_unknown_bounds=[average((generator_lo>hi)[mask],tt,te),average((generator_hi>lo)[mask],tt,te)]))
    summaries={}
    for group in [*sorted(set(types)),'combined']:
        mask=types==group if group!='combined' else np.ones(len(types),bool)
        summaries[str(group)]={'type_equal':describe(mask,True)}
        if group=='combined':summaries[str(group)]['controller_equal']=describe(mask,False)
    full=np.all(labels!=-2,axis=(2,3));supported=int(full.sum())
    # Full two-seed support needs a separate cross-fitted computation; never
    # relabel the common-seed in-sample statistic as cross-fitted evidence.
    cf=dict(status='UNAVAILABLE_GENERATOR_SEED1' if supported==0 else 'REQUIRES_TWO_SEED_SUPPLEMENT',
        full_menu_two_seed_controller_states=supported,total_controller_states=int(full.size),gain_pp=None,
        reason='Original training plan observes generated actions only at seed0; two-seed anchors alone do not identify the full-menu cross-fitted oracle')
    target=HERE/('PILOT_REPORT.json' if which=='pilot' else 'SELECTABLE_SPACE.json');assert not target.exists()
    write(target,dict(utc=utc(),evidence_level='DEVELOPMENT',summaries=summaries,full_state_crossfit_oracle=cf,
        common_comparison_seed=0,model_used=False,generator_seed1_not_imputed=True,
        interpretation='In-sample selectable space is optimistic and descriptive; no cross-fitted claim without complete paired seeds'))
    receipt_name='PILOT_REPORT.receipt.json' if which=='pilot' else 'SELECTABLE_SPACE.receipt.json'
    seal(receipt_name,[target,receipt,HERE/'headroom.py'],model_used=False)
    print('Model-free',which,'selectable-space report sealed; crossfit status:',cf['status'],flush=True)


if __name__=='__main__':main(sys.argv[1])
