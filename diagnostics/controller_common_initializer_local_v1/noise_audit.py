"""Reuse labels/predictions to audit the failed source learning gate."""
import numpy as np
from .study import HERE,BASE,FOLDS,SEEDS,read,write
from .review import center,ci


def run():
    d=np.load(HERE/'dataset.npz');fixed=np.load(HERE/'independent_static_actions.npz')['vector_motion']
    uid=np.load(HERE/'motion_raw.npz')['vector_motion'];rows=[];boot={}
    for fold,held in FOLDS.items():
        models=[np.load(HERE/'fits'/fold/'motion_complete'/f'seed{s}'/'predictions.npz')['probability'].astype(float) for s in SEEDS]
        p=np.mean(models,axis=0)
        groups=[(sc,sp) for sc in FOLDS.values() for sp in ('train','validation')]
        for sc,sp in groups:
            ix=np.flatnonzero((d['scene']==sc)&(d['split']==sp));ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
            n=len(ix)//4;no=int(d['entity_agent_mask'][ix[0]].sum())*2;nt=2 if sp=='train' else 4
            y=fixed[ix][:,[0,5,10,15],:,:no].mean(2).reshape(n,4,4,no)
            z=uid[ix][:,[0,5,10,15],:nt,:no].mean(2).reshape(n,4,4,no)
            pred=p[ix][:,[0,5,10,15],:no].reshape(n,4,4,no)
            ib=np.random.default_rng(2026101040+n).integers(n,size=(20000,n))
            for name,axes in [('state_eta',(0,2)),('state_controller_eta',(0,1,2))]:
                pc=center(pred,axes);yc=center(y,axes);zc=center(z,axes)
                bs=[];bc=[];btruth=[]
                for start in range(0,len(ib),500):
                    jj=ib[start:start+500];ax=tuple(a+1 for a in axes)
                    pp=center(pred[jj],ax);yy=center(y[jj],ax);zz=center(z[jj],ax)
                    bs.append((2*pp*yy-pp*pp).mean((1,2,3,4)))
                    bc.append((2*pp*zz-pp*pp).mean((1,2,3,4)))
                    btruth.append((2*yy*zz-yy*yy).mean((1,2,3,4)))
                bs,bc,btruth=map(np.concatenate,(bs,bc,btruth))
                tag='__'.join((fold,sc,sp,name))
                boot[tag+'__same_key_skill']=bs;boot[tag+'__uid_key_skill']=bc;boot[tag+'__label_panel_skill']=btruth
                rows.append(dict(fold=fold,scene=sc,split=sp,held=sc==held,metric=name,physical_states=n,
                    same_key_skill=float(np.mean(2*pc*yc-pc*pc)),same_key_CI95=ci(bs),
                    uid_key_skill=float(np.mean(2*pc*zc-pc*pc)),uid_key_CI95=ci(bc),
                    label_panel_skill=float(np.mean(2*yc*zc-yc*yc)),label_panel_CI95=ci(btruth),
                    same_panel_truth_power=float(np.mean(yc*yc)),cross_panel_truth_covariance=float(np.mean(yc*zc)),
                    same_vs_UID_truth_corr=float(np.corrcoef(yc.ravel(),zc.ravel())[0,1])))
    write(HERE/'noise_audit_results.json',dict(rows=rows,new_fit=0,new_query=0,new_rollout=0,
        prior_reuse='Stage30 and35 used native law. This exact same-vs-independent-noise evaluation has not been done for common-law changed targets; all newstage37predictions/labels reused.',
        interpretation='If even exact four-key labels predict UID interaction worse than zero, the finite-key target panel itself is not transferable to independent stochastic evaluation. Does not demonstrate learning potential, longQcause, or remove geometry as a competing limitation. No new data yet.'))
    np.savez_compressed(HERE/'noise_audit_bootstrap.npz',**boot)
    for r in rows:
        if r['metric']=='state_eta' and (r['held'] or r['scene']=='ring_exchange'):
            print(r['fold'],r['scene'],r['split'],r['same_key_CI95'],r['uid_key_CI95'],r['label_panel_CI95'])


if __name__=='__main__':run()
