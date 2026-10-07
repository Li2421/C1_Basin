"""Preregistered independent-block reproducibility analysis.

Old fitted predictions/checkpoints are fixed before opening new continuation
outcomes. B15 is always a named block of16, not a fabricated Q48 certification.
Numerical results remain unknown and never become task failures.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr,spearmanr,chi2
from shared_rollout_db.src.rollout_db import connect,canonical
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import read,write,csvwrite

OUT=Path(__file__).resolve().parent
SRC=OUT.parent/'orthoflow3_state_context_learning_audit_v1'
REP=OUT/'seed_replication'
ETAS=(10,15)

def correlation(a,b):
    a=np.asarray(a);b=np.asarray(b);ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3 or np.std(a[ok])<1e-12 or np.std(b[ok])<1e-12:return None
    return dict(Pearson=float(pearsonr(a[ok],b[ok]).statistic),Spearman=float(spearmanr(a[ok],b[ok]).statistic),n=int(ok.sum()))

def load():
    states=read(REP/'states.json');pairs=read(REP/'pairs.json');profiles=read(REP/'protocol.json')['profiles']
    lookup={(p['state_index'],p['eta_index']):p for p in pairs}
    y=np.full((2,len(states),2,64),np.nan);keys=[]
    with connect(True) as db:
      for ci,c in enumerate(profiles):
       for si,s in enumerate(states):
        for ei,e in enumerate(ETAS):
            p=lookup[(si,e)]
            rows=db.execute('SELECT rollout_uid,seed_key,success,numerical_failure,conflict_quarantined,compatibility_quality FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(s['uid'],p['eta_uid'],c['controller_uid'])).fetchall()
            seen=set()
            for r in rows:
                k=json.loads(r['seed_key'])['future_index']
                if k>=64:continue
                assert not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE'
                assert k not in seen;seen.add(k)
                if not r['numerical_failure']:y[ci,si,ei,k]=r['success']
                keys.append(dict(rollout_uid=r['rollout_uid'],controller=ci,state=si,eta=e,seed=k,numerical=bool(r['numerical_failure'])))
            assert seen==set(range(64)),(s['uid'],ci,e,len(seen))
    return states,y,keys

def block_b15(y):
    s=np.nansum(y,axis=-1);f=np.nansum(1-y,axis=-1)
    return s>=15,(f>=2),~((s>=15)|(f>=2))

def main():
    states,y,keys=load();old=np.nanmean(y[...,:16],axis=-1);fresh=np.nanmean(y[...,16:],axis=-1)
    rng=np.random.default_rng(2026100421)
    cells=[];contrast=[]
    for c in range(2):
      for s,state in enumerate(states):
        for e,ei in enumerate(ETAS):
            vals=y[c,s,e];n=np.isfinite(vals[16:]).sum();q=fresh[c,s,e]
            denom=1+1.96**2/n;center=(q+1.96**2/(2*n))/denom
            half=1.96*np.sqrt(q*(1-q)/n+1.96**2/(4*n*n))/denom
            cells.append(dict(controller=c,state_uid=state['uid'],source_group=state['source_group'],eta_index=ei,
                old_Q16=float(old[c,s,e]),fresh_Q48=float(q),fresh_n=int(n),fresh_Wilson_lower=float(center-half),fresh_Wilson_upper=float(center+half),
                **{f'block{b}_success':int(np.nansum(vals[b*16:(b+1)*16])) for b in range(4)},
                **{f'block{b}_numerical':int(np.isnan(vals[b*16:(b+1)*16]).sum()) for b in range(4)}))
        v=y[c,s,0,16:]-y[c,s,1,16:];v=v[np.isfinite(v)]
        boot=v[rng.integers(len(v),size=(4000,len(v)))].mean(1);lo,hi=np.quantile(boot,[.025,.975])
        delta=float(fresh[c,s,0]-fresh[c,s,1]);od=float(old[c,s,0]-old[c,s,1])
        contrast.append(dict(controller=c,state_uid=state['uid'],old_delta=od,fresh_delta=delta,
            fresh_paired_ci_lower=float(lo),fresh_paired_ci_upper=float(hi),
            reliable_fresh_sign=1 if lo>0 else -1 if hi<0 else 0,
            old_strong=abs(od)>=.25,old_sign_replicates=bool(od*delta>0)))
    csvwrite(OUT/'seed_replication_cells.csv',cells);csvwrite(OUT/'seed_replication_contrasts.csv',contrast)
    write(OUT/'seed_replication_DB_keys.json',keys)
    blocks=np.stack([np.nanmean(y[...,b*16:(b+1)*16],axis=-1) for b in range(4)])
    reliability=[]
    for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
        # Remove controller and eta main effects: test state-specific residual reproducibility.
        aa=blocks[a]-blocks[a].mean(axis=1,keepdims=True)
        bb=blocks[b]-blocks[b].mean(axis=1,keepdims=True)
        reliability.append(dict(block_a=a,block_b=b,centered_state_Q_correlation=correlation(aa.ravel(),bb.ravel()),
            eta_contrast_correlation=correlation((blocks[a,:,:,0]-blocks[a,:,:,1]).ravel(),(blocks[b,:,:,0]-blocks[b,:,:,1]).ravel())))
    centered_old=old-old.mean(axis=1,keepdims=True);centered_fresh=fresh-fresh.mean(axis=1,keepdims=True)
    # Conditional null: one p per controller+eta, no state dependence. All
    # iterations have the same finite n pattern; no asymptotic low-count test.
    s=np.nansum(y[...,16:],axis=-1);n=np.isfinite(y[...,16:]).sum(axis=-1)
    pooled=s.sum(axis=1,keepdims=True)/n.sum(axis=1,keepdims=True)
    observed=float(np.sum(n*(s/n-pooled)**2))
    sim=rng.binomial(n[None],pooled[None],size=(5000,*n.shape));rates=sim/n
    pm=sim.sum(axis=2,keepdims=True)/n.sum(axis=1,keepdims=True)
    null=np.sum(n[None]*(rates-pm)**2,axis=(1,2,3))
    strong=[r for r in contrast if r['old_strong']]
    summary=dict(states=len(states),controllers=2,eta_count=2,old_seed_block=[0,15],fresh_seed_blocks=[[16,31],[32,47],[48,63]],
        valid_new=int(np.isfinite(y[...,16:]).sum()),numerical_new=int(np.isnan(y[...,16:]).sum()),
        raw_Q_correlation=correlation(old.ravel(),fresh.ravel()),
        state_residual_old_to_fresh=correlation(centered_old.ravel(),centered_fresh.ravel()),
        eta_contrast_old_to_fresh=correlation((old[:,:,0]-old[:,:,1]).ravel(),(fresh[:,:,0]-fresh[:,:,1]).ravel()),
        old_strong_contrasts=len(strong),old_strong_sign_replicated=sum(r['old_sign_replicates'] for r in strong),
        fresh_sign_certified_positive=sum(r['reliable_fresh_sign']==1 for r in contrast),
        fresh_sign_certified_negative=sum(r['reliable_fresh_sign']==-1 for r in contrast),
        state_homogeneity_null=dict(statistic=observed,Monte_Carlo_p=float((1+np.sum(null>=observed))/5001),null_samples=5000),
        block_reliability=reliability,
        qualification='Source diagnostic eta panel selected earlier on source data; independent seeds test repeatability, not unseen-state/controller generalization.')
    write(OUT/'seed_reproducibility_summary.json',summary)
    evaluate_models(states,y)
    print(json.dumps(summary,indent=2))

def evaluate_models(states,y):
    olds=read(SRC/'source_states.json');lookup={s['uid']:i for i,s in enumerate(states)}
    paths=[(SRC,k) for k in ('eta_only','full_raw','wide_db_eta_only','wide_db_full_raw','wide_db_full_scaled')]
    role=OUT/'role_control'
    if all((role/'models'/k/'fold2_seed41/complete.json').exists() for k in ('wide_db_role_zero','wide_db_role_aware')):
        paths.extend((role,k) for k in ('wide_db_role_zero','wide_db_role_aware'))
    response=OUT/'response_control'
    if all((response/'models'/k/'fold2_seed41/complete.json').exists() for k in ('response_zero','response_mean','response_entity')):
        paths.extend((response,k) for k in ('response_zero','response_mean','response_entity'))
    result=[];decisions=[]
    for root,kind in paths:
     for seed in (17,23,41):
        z=np.full(y.shape[:3],np.nan)
        for fold in range(3):
            p=np.load(root/'models'/kind/f'fold{fold}_seed{seed}'/'predictions.npz')
            for j,i in enumerate(p['indices']):
                state=olds[int(i)//16]['uid'];e=ETAS.index(int(i)%16)
                if state in lookup:z[:,lookup[state],e]=p['best_correct'][:,j]
        assert np.isfinite(z).all()
        s=np.nansum(y[...,16:],axis=-1);f=np.nansum(1-y[...,16:],axis=-1)
        q=s/(s+f);prob=expit(z);choices=z.argmax(axis=-1)
        sr=[];oracle=[];eligsel=[];reg=[];unknown=[]
        for c in range(2):
         for i,state in enumerate(states):
            e=choices[c,i]
            row=dict(kind=kind,seed=seed,controller=c,state_uid=state['uid'],eta_index=ETAS[e],
                predicted=float(prob[c,i,e]),fresh_Q48=float(q[c,i,e]),fresh_oracle_Q48=float(q[c,i].max()))
            reg.append(float(q[c,i].max()-q[c,i,e]))
            for b in range(1,4):
                good,bad,unk=block_b15(y[c,i,:,b*16:(b+1)*16]);av=bool(good.any())
                sr.append(bool(good[e]));oracle.append(av);unknown.append(bool(unk[e]));eligsel.append(bool(good[e] and av))
                row[f'B15_block{b}']=bool(good[e]);row[f'oracle_block{b}']=av
            decisions.append(row)
        result.append(dict(kind=kind,seed=seed,fresh_NLL=float(np.mean(np.sum(s*np.logaddexp(0,-z)+f*np.logaddexp(0,z),axis=(1,2))/np.sum(s+f,axis=(1,2)))),
            fresh_MAE=float(abs(prob-q).mean()),fresh_selected_Q48=float(np.mean([q[c,i,choices[c,i]] for c in range(2) for i in range(len(states))])),
            fresh_regret=float(np.mean(reg)),fresh_B15_blocks=int(sum(sr)),fresh_oracle_B15_blocks=int(sum(oracle)),
            fresh_unknown_selected_blocks=int(sum(unknown)),case_blocks=len(sr),
            fresh_state_eta_contrast_correlation=correlation((prob[:,:,0]-prob[:,:,1]).ravel(),(q[:,:,0]-q[:,:,1]).ravel())['Pearson'] if correlation((prob[:,:,0]-prob[:,:,1]).ravel(),(q[:,:,0]-q[:,:,1]).ravel()) else None))
    csvwrite(OUT/'fresh_seed_frozen_model_metrics.csv',result);csvwrite(OUT/'fresh_seed_frozen_model_decisions.csv',decisions)

if __name__=='__main__':main()
