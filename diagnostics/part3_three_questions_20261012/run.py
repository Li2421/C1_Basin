"""Three bounded questions, existing P1 labels, joint held controller/state tests."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from diagnostics.part3_transfer_feasibility_round3_20261012 import ablate as old
HERE=Path(__file__).resolve().parent
BWS=(.5,1.,2.)
REGS=(.0001,.001,.01)

def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def indices(cs,ss):
    return np.array([c*24+s for c in cs for s in ss],int)

def transform(parts,train,test,arm):
    a=[];b=[]
    for i,part in enumerate(parts if arm=='R5' else parts[:1]):
        mean=part[train].mean(0)
        std=part[train].std(0)
        if i==0:
            scale=np.maximum(std,.05)*np.sqrt(part.shape[1])
        else:
            scale=max(float(np.sqrt(np.mean(std**2))),.05)*np.sqrt(6*part.shape[1])
        a.append((part[train]-mean)/scale)
        b.append((part[test]-mean)/scale)
    return np.concatenate(a,1),np.concatenate(b,1)

def grid_predictions(parts,q,train,test,arm):
    x,v=transform(parts,train,test,arm)
    dt=cdist(x,x,'sqeuclidean');dv=cdist(v,x,'sqeuclidean')
    mean=q[train].mean(0);yc=q[train]-mean
    for bw in BWS:
        kernel=np.exp(-dt/(2*bw*bw))
        ev,vec=eigh(kernel,check_finite=False)
        ev=np.maximum(ev,0)
        target=vec.T@yc
        kv=np.exp(-dv/(2*bw*bw))@vec
        for reg in REGS:
            pred=mean+kv@(target/(ev[:,None]+len(train)*reg))
            yield (bw,reg),np.clip(pred,0,1)

def choose(parts,q,sources,states,arm):
    configs={(bw,reg):[] for bw in BWS for reg in REGS}
    for fold in range(3):
        ic=[c for j,c in enumerate(sources) if j%3!=fold]
        vc=[c for j,c in enumerate(sources) if j%3==fold]
        it=[s for j,s in enumerate(states) if j%3!=fold]
        vs=[s for j,s in enumerate(states) if j%3==fold]
        train=indices(ic,it);test=indices(vc,vs)
        for key,pred in grid_predictions(parts,q,train,test,arm):
            configs[key].append(float(np.mean((pred-q[test])**2)))
    scores={k:float(np.mean(v)) for k,v in configs.items()}
    best=min(scores,key=lambda k:(scores[k],k))
    return best,[dict(bandwidth=k[0],regularization=k[1],inner_mse=v) for k,v in scores.items()]

def fit_predict(parts,q,train,test,arm,key,donors=None):
    x,v=transform(parts,train,test,arm)
    bw,reg=key
    kernel=np.exp(-cdist(x,x,'sqeuclidean')/(2*bw*bw))
    ev,vec=eigh(kernel,check_finite=False);ev=np.maximum(ev,0)
    mean=q[train].mean(0)
    coef=vec@((vec.T@(q[train]-mean))/(ev[:,None]+len(train)*reg))
    pred=np.clip(mean+np.exp(-cdist(v,x,'sqeuclidean')/(2*bw*bw))@coef,0,1)
    mismatch=[]
    if donors is not None:
        for donor in donors:
            donor_rows=donor*24+test%24
            _,d=transform(parts,train,donor_rows,arm)
            # Physical states of all controllers are identical and held fixed.
            assert np.array_equal(v[:,:24],d[:,:24])
            mismatch.append(np.clip(mean+np.exp(-cdist(d,x,'sqeuclidean')/(2*bw*bw))@coef,0,1))
    return pred,np.array(mismatch),coef,mean

def value(a,sel):
    return np.take_along_axis(a,sel[...,None],axis=-1)[...,0]

def gain(sel,ref,lo,hi):
    return np.where(sel==ref,0,value(lo,sel)-value(hi,ref)),np.where(sel==ref,0,value(hi,sel)-value(lo,ref))

def boot(a):
    a=np.asarray(a,float);ix=np.random.default_rng(20261012).integers(len(a),size=(20000,len(a)))
    return np.quantile(a[ix].mean(1),[.025,.975]).tolist()

def aggregate_interval(a,b):
    return [float(np.mean(a)),float(np.mean(b))]

def main():
    start=time.monotonic()
    names,blocks,y,u,lo,hi,mid=old.load()
    p=json.loads((old.P1/'P1_PROTOCOL.json').read_text())
    state=np.array([np.concatenate([np.array(s['physical'][k]).ravel() for k in ('positions','velocities','goals')]) for s in p['states']])
    assert state.shape==(24,24)
    parts=[np.broadcast_to(state,(14,24,24)).reshape(336,24)]+[a.reshape(336,-1) for a in blocks['R5']]
    q=mid.reshape(336,16)
    pred=np.full((2,14,24,16),np.nan)
    mismatch=np.full((14,13,24,16),np.nan)
    fixed=np.full((14,24),-1,int)
    folds=[]
    (HERE/'models').mkdir(exist_ok=True)
    for held in range(14):
        sources=[c for c in range(14) if c!=held]
        for fold in range(4):
            trainstates=[s for s in range(24) if s%4!=fold]
            teststates=[s for s in range(24) if s%4==fold]
            train=indices(sources,trainstates);test=indices([held],teststates)
            assert held not in (train//24) and not set(train%24)&set(test%24)
            eta=int(np.argmax(q[train].mean(0)));fixed[held,teststates]=eta
            row=dict(held=names[held],state_fold=fold,train_controllers=sources,train_states=trainstates,test_states=teststates,source_fixed_eta=eta,models={})
            for ai,arm in enumerate(('R0','R5')):
                key,scores=choose(parts,q,sources,trainstates,arm)
                pp,mm,coef,mean=fit_predict(parts,q,train,test,arm,key,sources if arm=='R5' else None)
                pred[ai,held,teststates]=pp
                if arm=='R5':
                    for donor in range(13):mismatch[held,donor,teststates]=mm[donor]
                np.savez_compressed(HERE/'models'/f'c{held}_s{fold}_{arm}.npz',train_rows=train,coefficient=coef,intercept=mean,bandwidth=key[0],regularization=key[1])
                row['models'][arm]=dict(bandwidth=key[0],regularization=key[1],inner_scores=scores)
            folds.append(row)
        print(json.dumps(dict(completed_controller=names[held],elapsed_seconds=round(time.monotonic()-start,2))),flush=True)
    assert np.isfinite(pred).all() and np.isfinite(mismatch).all()
    np.savez_compressed(HERE/'predictions.npz',prediction=pred,mismatched=mismatch,fixed_eta=fixed,names=np.array(names),state_uids=np.array([s['state_uid'] for s in p['states']]))
    write('folds.json',folds)
    rows=[]
    for c,name in enumerate(names):
        r=dict(controller=name,unknown_seeds=int(u[c].sum()),source_fixed_Q=aggregate_interval(value(lo[c],fixed[c]),value(hi[c],fixed[c])),models={})
        for ai,arm in enumerate(('R0','R5')):
            pp=pred[ai,c];sel=np.argmax(pp,-1)
            gl,gh=gain(sel,fixed[c],lo[c],hi[c]);bl,bh=gain(sel,np.argmax(pred[0,c],-1),lo[c],hi[c])
            ranks=[float(spearmanr(pp[s],mid[c,s]).statistic) for s in range(24) if np.std(mid[c,s])>0 and np.std(pp[s])>0]
            residual=(pp-pp.mean(-1,keepdims=True))-(mid[c]-mid[c].mean(-1,keepdims=True))
            ap=pp-value(pp,fixed[c])[:,None];aq=mid[c]-value(mid[c],fixed[c])[:,None]
            minerr=np.maximum(0,np.maximum(lo[c]-pp,pp-hi[c]))**2
            maxerr=np.maximum((lo[c]-pp)**2,(hi[c]-pp)**2)
            # Exact empirical max regret interval, sharing the selected cell.
            rl=[];rh=[]
            for s,k in enumerate(sel):
                oth=[j for j in range(16) if j!=k]
                rl.append(max(0,float(lo[c,s,oth].max()-hi[c,s,k])))
                rh.append(max(0,float(hi[c,s,oth].max()-lo[c,s,k])))
            bins=[]
            for b in range(5):
                mask=(pp>=b/5)&((pp<(b+1)/5) if b<4 else (pp<=1))
                if np.any(mask):bins.append(dict(bin=b,n=int(mask.sum()),prediction_mean=float(pp[mask].mean()),Q_interval=[float(lo[c][mask].mean()),float(hi[c][mask].mean())]))
            r['models'][arm]=dict(MSE=float(np.mean((pp-mid[c])**2)),MSE_unknown_interval=aggregate_interval(minerr,maxerr),eta_centered_MSE=float(np.mean(residual**2)),advantage_MSE=float(np.mean((ap-aq)**2)),rank=float(np.mean(ranks)) if ranks else None,top1_Q=aggregate_interval(value(lo[c],sel),value(hi[c],sel)),gain_fixed=aggregate_interval(gl,gh),gain_blind=aggregate_interval(bl,bh),empirical_oracle_regret=aggregate_interval(rl,rh),selected_eta=sel.tolist(),calibration=bins)
        r['mismatched_R5_MSE']=float(np.mean((mismatch[c]-mid[c][None])**2))
        rows.append(r)
    summary={}
    for arm in ('R0','R5'):
        metrics=[r['models'][arm] for r in rows]
        sm={k:float(np.mean([v[k] for v in metrics])) for k in ('MSE','eta_centered_MSE','advantage_MSE','rank')}
        sm.update({k:np.mean([v[k] for v in metrics],axis=0).tolist() for k in ('MSE_unknown_interval','top1_Q','gain_fixed','gain_blind','empirical_oracle_regret')})
        for base in ('fixed','blind'):
            lower=[v['gain_'+base][0] for v in metrics]
            sm[base+'_nonnegative_controllers']=int(np.sum(np.array(lower)>=0))
            sm[base+'_lower_gain_bootstrap95']=boot(lower)
        summary[arm]=sm
    delta_mid=np.array([r['models']['R0']['MSE']-r['models']['R5']['MSE'] for r in rows])
    dl=(pred[0]-lo)**2-(pred[1]-lo)**2
    dh=(pred[0]-hi)**2-(pred[1]-hi)**2
    mismatch_delta=[r['mismatched_R5_MSE']-r['models']['R5']['MSE'] for r in rows]
    criterion=summary['R5']
    passed=all(criterion['gain_'+k][0]>0 and criterion[k+'_nonnegative_controllers']>=10 and criterion[k+'_lower_gain_bootstrap95'][0]>0 for k in ('fixed','blind'))
    out=dict(scope='Opened P1 real Ring FF checkpoint inventory; joint held controller and held state; exploratory, not new-label confirmation',new_rollouts=0,new_features=0,reused_seed_labels=int(y.size),unknown_seeds=int(u.sum()),summary=summary,source_fixed_Q=np.mean([r['source_fixed_Q'] for r in rows],axis=0).tolist(),Q_MSE_reduction=dict(mean=float(delta_mid.mean()),descriptive_controller_bootstrap95=boot(delta_mid),unknown_safe_interval=aggregate_interval(np.minimum(dl,dh),np.maximum(dl,dh)),positive_controllers=int(np.sum(delta_mid>0))),matched_descriptor_MSE_reduction=dict(mean=float(np.mean(mismatch_delta)),descriptive_controller_bootstrap95=boot(mismatch_delta),positive_controllers=int(np.sum(np.array(mismatch_delta)>0))),selection_verdict='SUPPORTED_IN_THIS_PANEL' if passed else 'NOT_VALIDATED',per_controller=rows,total_seconds=time.monotonic()-start)
    write('results.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='per_controller'},indent=2),flush=True)

if __name__=='__main__':main()
