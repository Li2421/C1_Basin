"""Local transfer, independent-noise evaluation and state-level controls."""
import argparse
import numpy as np
from .study import HERE,EVAL,FOLDS,ARMS,SEEDS,read,write,sha,freeze


def center(x,axes):
    for axis in axes:x=x-x.mean(axis,keepdims=True)
    return x


def ci(x):return np.quantile(x,[.025,.975]).tolist()


def review(fold):
    d=dict(np.load(HERE/'dataset.npz'));raw=np.load(EVAL/'motion_raw.npz')['vector_motion']
    ensemble={};models={}
    for arm in ARMS:
        models[arm]=[]
        for seed in SEEDS:
            f=HERE/'fits'/fold/arm/f'seed{seed}'
            assert (f/'complete.json').exists()
            p=np.load(f/'predictions.npz');np.testing.assert_array_equal(p['state_uid'],d['state_uid'])
            models[arm].append(p['probability'][:,[0,5,10,15]].astype(np.float64))
        ensemble[arm]=np.mean(models[arm],axis=0)
    rows=[];boot={};predictions={};failures=[]
    groups=[(sc,'source_validation',d['split']=='validation') for sc in sorted(set(d['scene'])) if sc!=FOLDS[fold]]
    groups.extend((FOLDS[fold],name,mask) for name,mask in
        [('held_all',np.ones(len(d['scene']),bool)),('held_train_origin',d['split']=='train'),('held_val_origin',d['split']=='validation')])
    for sc,label,split in groups:
        ix=np.flatnonzero((d['scene']==sc)&split)
        ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
        n=len(ix)//4;no=int(d['entity_agent_mask'][ix[0]].sum())*2
        sid=d['state_uid'][ix][::4];half=np.empty((len(ix),4,2,no))
        for r,i in enumerate(ix):
            nt=2 if d['split'][i]=='train' else 4
            a=raw[i][[0,5,10,15],:nt,:no]
            assert np.isfinite(a).all()
            half[r,:,0]=a[:,:nt//2].mean(1);half[r,:,1]=a[:,nt//2:].mean(1)
        half=half.reshape(n,4,4,2,no);y=half.mean(3)
        pp={arm:p[ix,:,:no].reshape(n,4,4,no) for arm,p in ensemble.items()}
        ib=np.random.default_rng(2026101040+n).integers(n,size=(20000,n))
        w=np.array([np.bincount(z,minlength=n) for z in ib],float)
        tag=str(sc)+'__'+label;boot[tag+'__indices']=ib
        predictions[tag+'__states']=sid;predictions[tag+'__target_halves']=half
        for arm,p in pp.items():predictions[tag+'__'+arm]=p
        # Exact mean over all derangements. Recompute the donor distribution
        # inside each physical-state bootstrap, including repeated draws.
        p=pp['motion_complete'];flat_p=p.reshape(n,-1);flat_y=y.reshape(n,-1)
        distance=(flat_y**2).mean(1)[:,None]+(flat_p**2).mean(1)[None,:]-2*flat_y@flat_p.T/flat_p.shape[1]
        match=np.diag(distance)
        expected_wrong=(distance.sum()-match.sum())/(n*(n-1))
        bmatch=w@match/n
        bwrong=(np.einsum('bi,ij,bj->b',w,distance,w)-w@match)/(n*(n-1))
        b=bwrong-bmatch;boot[tag+'__state_mismatch_gain']=b
        eta_error=((pp['motion_eta_only']-y)**2).mean((1,2,3));b_eta=w@(eta_error-match)/n
        boot[tag+'__eta_MSE_gain']=b_eta
        # Exact mean over all nonidentical controller donors at the SAME state.
        cf=p.reshape(n,4,-1);cy=y.reshape(n,4,-1)
        dc=(cy**2).mean(-1)[:,:,None]+(cf**2).mean(-1)[:,None,:]-2*np.einsum('ncd,nkd->nck',cy,cf)/cf.shape[-1]
        cw=(dc.sum((1,2))-np.trace(dc,axis1=1,axis2=2))/12
        bc=w@(cw-match)/n;boot[tag+'__controller_mismatch_gain']=bc
        rows.append(dict(fold=fold,scene=str(sc),split=label,physical_states=n,metric='full_correction_MSE',
            matched=float(match.mean()),eta_only=float(eta_error.mean()),state_mismatched=float(expected_wrong),
            correct_state_gain=float(expected_wrong-match.mean()),correct_state_CI95=ci(b),
            eta_only_MSE_gain=float((eta_error-match).mean()),eta_only_MSE_gain_CI95=ci(b_eta),
            correct_controller_gain=float((cw-match).mean()),correct_controller_CI95=ci(bc)))
        for name,axes in [('state_eta',(0,2)),('state_controller_eta',(0,1,2))]:
            pc=center(p,axes);tc=center(y,axes);a=center(half[:,:,:,0],axes);z=center(half[:,:,:,1],axes)
            skill=float(np.mean(2*pc*tc-pc*pc));power=float(np.mean(a*z))
            sb=[];pb=[];ec=[]
            for start in range(0,len(ib),250):
                jj=ib[start:start+250];ax=tuple(x+1 for x in axes)
                cp=center(p[jj],ax);yy=center(y[jj],ax)
                ha=center(half[jj,:,:,0],ax);hb=center(half[jj,:,:,1],ax)
                sb.append(np.mean(2*cp*yy-cp*cp,axis=(1,2,3,4)))
                pb.append(np.mean(ha*hb,axis=(1,2,3,4)))
                ec.append(np.mean((cp-yy)**2,axis=(1,2,3,4)))
            sb=np.concatenate(sb);pb=np.concatenate(pb);ec=np.concatenate(ec)
            boot[tag+'__'+name+'__skill']=sb;boot[tag+'__'+name+'__true_energy']=pb
            per_seed=[]
            for pred in models['motion_complete']:
                v=center(pred[ix,:,:no].reshape(n,4,4,no),axes)
                per_seed.append(float(np.mean(2*v*tc-v*v)))
            null=center(pp['motion_eta_only'],axes)
            assert np.max(abs(null))<1e-6
            rows.append(dict(fold=fold,scene=str(sc),split=label,physical_states=n,metric=name,
                interaction_skill=skill,skill_CI95=ci(sb),true_interaction_energy=power,true_energy_CI95=ci(pb),
                per_seed_skill=per_seed,correlation=float(np.corrcoef(pc.ravel(),tc.ravel())[0,1]) if min(pc.std(),tc.std())>1e-15 else None,
                null_eta_interaction_max=float(np.max(abs(null)))))
        if label=='held_all':
            for i in np.argsort(match)[-5:][::-1]:failures.append(dict(fold=fold,state_uid=str(sid[i]),MSE=float(match[i]),eta_MSE=float(eta_error[i])))
    np.savez_compressed(HERE/f'bootstrap_{fold}.npz',**boot)
    np.savez_compressed(HERE/f'predictions_{fold}.npz',**predictions)
    write(HERE/f'review_{fold}.json',dict(rows=rows,failures=failures,root_cause_closed=False,
        deterministic_target=True, stored_copies_are_not_independent_seeds=True, scope='Actual correction target before FF integration; deterministic exact target, not rollout-half reliability. NEVER terminal-Q or selection-success results. Full physical-state/controller-descriptor bundle permutation holds controller program fixed; not isolation of explicit S channels from state-dependent C fingerprints.',
        permutation='Exact average over every non-self donor (equal to expectation over uniformly random derangements). Entire donor pool reweighted inside each physical-state bootstrap. Controller donors vary at same physicalstate.',
        calibration='SourceVAL used for checkpoint selection. Held data are opened DEVELOPMENT; independent physicalscene excluded from all fitting/normalization/checkpoint choices.',
        new_rollouts=0,code_sha256=sha(__file__)))
    print(fold,[(r['metric'],r.get('skill_CI95',r.get('correct_state_CI95'))) for r in rows if r['split']=='held_all'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,choices=FOLDS);a=p.parse_args();review(a.fold)
