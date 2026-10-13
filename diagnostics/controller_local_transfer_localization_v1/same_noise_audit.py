"""Falsify the independent-noise explanation for the local LOSO failure.

Use all four cached TRAIN noise keys, but on the held physical scene. This is
a conditional finite-noise diagnostic, not population-noise confirmation.
"""
import numpy as np
from .study import HERE,EVAL,FOLDS,SEEDS,write
from .review import center,ci


def run():
    d=np.load(HERE/'dataset.npz');a=np.load(EVAL/'independent_static_actions.npz')['vector_motion']
    rows=[];boot={}
    for fold,sc in FOLDS.items():
        ix=np.flatnonzero(d['scene']==sc);ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
        n=len(ix)//4;no=int(d['entity_agent_mask'][ix[0]].sum())*2
        y=a[ix][:,[0,5,10,15],:,:no].mean(2).reshape(n,4,4,no);assert np.isfinite(y).all()
        models=[np.load(HERE/'fits'/fold/'motion_complete'/f'seed{s}'/'predictions.npz')['probability'][ix][:,[0,5,10,15],:no].reshape(n,4,4,no).astype(float) for s in SEEDS]
        p=np.mean(models,axis=0);ib=np.random.default_rng(2026101040+n).integers(n,size=(20000,n))
        for name,axes in [('state_eta',(0,2)),('state_controller_eta',(0,1,2))]:
            pc=center(p,axes);yc=center(y,axes);sb=[]
            for start in range(0,len(ib),500):
                jj=ib[start:start+500];ax=tuple(z+1 for z in axes)
                pp=center(p[jj],ax);yy=center(y[jj],ax)
                sb.append(np.mean(2*pp*yy-pp*pp,axis=(1,2,3,4)))
            b=np.concatenate(sb);boot[fold+'__'+name]=b
            rows.append(dict(fold=fold,scene=sc,physical_states=n,metric=name,
                interaction_skill=float(np.mean(2*pc*yc-pc*pc)),CI95=ci(b),
                per_seed_skill=[float(np.mean(2*center(v,axes)*yc-center(v,axes)**2)) for v in models]))
    np.savez_compressed(HERE/'same_noise_bootstrap.npz',**boot)
    write(HERE/'same_noise_results.json',dict(rows=rows,new_fit=0,new_rollout=0,new_query=0,
        scope='Same four static noise keys as sourceTRAIN, but heldscene labels neverfit. CIs conditional onthisfixednoisepanel. ComparewithindependentUIDnoise primary; cannot substitute fornoise-populationconfirmation.'))
    print([(r['fold'],r['metric'],r['CI95']) for r in rows])


if __name__=='__main__':run()
