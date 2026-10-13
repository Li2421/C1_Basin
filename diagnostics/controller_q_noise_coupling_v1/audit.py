"""Censor-aware, paired-noise variance screen before any long-Q fits."""
import itertools
import numpy as np
from .study import HERE,ROOT,SCENE,FAMILIES,read,write,freeze,sha
from diagnostics.controller_flow_endpoint_probe_v1.analyze import energy
from shared_rollout_db.src.rollout_db import connect

def moment_variance(errors,mask,weights):
    # errors: panel,state,controller-or-pair,eta. Same pair is shared within S.
    # Remove S and eta additive terms separately for each controller.
    mask=mask.astype(float);count=mask.sum(-1)
    h=np.eye(mask.shape[-1])[None,None]*mask[...,None]
    h-=mask[...,None]*mask[...,None,:]/np.maximum(count,1)[...,None,None]
    he=np.einsum('ncef,pncf->pnce',h,errors)
    base_by_state=np.mean(he*he,axis=0).sum((1,2))
    cov=np.einsum('pnce,pncf->ncef',he,he)/len(errors)
    aa=np.einsum('bn,ncef->bcef',weights,h)
    inv=np.linalg.pinv(aa,rcond=1e-12)
    rhs=np.einsum('bn,pnce->bpce',weights,he)
    common_sub=np.einsum('bpce,bcef,bpcf->b',rhs,inv,rhs)/len(errors)
    independent_sub=np.einsum('bn,ncef,bcef->b',weights**2,cov,inv)
    base=np.einsum('bn,n->b',weights,base_by_state)
    denominator=np.einsum('bn,n->b',weights,count.sum(-1))
    return (base-common_sub)/denominator,(base-independent_sub)/denominator

def verify():
    # Exhaustive finite-panel verification, including missing cells and
    # duplicate physical states in a bootstrap sample.
    rng=np.random.default_rng(2031);z=rng.normal(size=(3,2,3,4));pairs=list(itertools.combinations(range(4),2))
    e=np.stack([z[...,list(p)].mean(-1)-z.mean(-1) for p in pairs])
    mask=np.ones((3,2,3),bool);mask[0,0,1]=False;w=np.array([2,1,1.])
    def direct(v):
        total=0.;nobs=0.
        for c in range(2):
            ii,jj=np.where(mask[:,c]);design=np.c_[np.eye(3)[ii],np.eye(3)[jj]]
            sw=np.sqrt(w[ii]);fit=np.linalg.lstsq(design*sw[:,None],v[ii,c,jj]*sw,rcond=1e-12)[0]
            total+=np.sum(w[ii]*(v[ii,c,jj]-design@fit)**2);nobs+=w[ii].sum()
        return total/nobs
    a,b=moment_variance(e,mask,w[None])
    ec=np.mean([direct(x) for x in e])
    ei=np.mean([direct(np.stack([e[p,s] for s,p in enumerate(choice)])) for choice in itertools.product(range(6),repeat=3)])
    np.testing.assert_allclose([a[0],b[0]],[ec,ei],rtol=0,atol=1e-12)

def run():
    verify()
    p=read(HERE/'protocol.json');cases=read(HERE/'cases.json')
    sids=sorted(s['state_uid'] for s in p['states']);lookup={s:i for i,s in enumerate(sids)}
    ci={s:i for i,s in enumerate(FAMILIES)};ei={j:i for i,j in enumerate(p['eta_indices'])}
    shape=(32,4,4,4);rs=np.zeros(shape);ru=np.zeros(shape);seen=np.zeros(shape,bool);uids=np.empty(shape,'<U80')
    safety=[];collisions=[];total=0.
    with connect(True) as db:
        for c in cases:
            r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            assert r is not None and not r['conflict_quarantined']
            assert all(r[k]==c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
            i,j,k=lookup[c['state_uid']],ci[c['controller_name']],ei[c['eta_index']]
            rs[i,j,k,c['seed']]=r['success'];ru[i,j,k,c['seed']]=r['numerical_failure'];seen[i,j,k,c['seed']]=True;uids[i,j,k,c['seed']]=r['rollout_uid']
            raw=read(HERE/'raw'/(c['rollout_uid']+'.json'));total+=raw['elapsed_seconds']
            if r['numerical_failure']:safety.append(dict(case=c,error=raw['error']))
            if r['collision']:collisions.append(dict(case=c,summary=raw['summary']))
    assert seen.all()
    freeze(HERE/'variance_audit_protocol.json',dict(
        prior_gate=p['cause_test']['gate_before_training'],
        censor='Seven numerical outcomes found. Finite-panel variance uses cells with ALL4known outcomes; same mask for bothcouplings/all6subsets, no wholephysicalstate dropped, no selection bysuccess. Reportexcludedcells. True-interaction power uses ALL original independent-UID labels with full censorintervals.',
        projection='Weighted least-squares S+eta nuisance projection perC; bootstrap repeats wholephysicalstates. Fortriple, all6C contrasts projected identically; missing contrast cells excluded symmetrically. Exact6subset expectation; independent subsets perphysicalstate retain withinstateC/eta covariance.',
        uncertainty='20000physicalstatebootstrap. Common-noise variance CI conditionalonfour shared noise streams; originalUID true-energy CI uses independent noise acrossphysicalstates. Neither source32states nor thisdata-dependent methoddevelopment is finalconfirmation.',
        identity='Moment formula checked against exhaustive6^3subset assignments withmissingcells andduplicatephysicalstateweights.',
        code_sha256=sha(__file__)))
    # Original stochastic execution has the same marginal controller/noise law,
    # but independent per-state streams; reuse its true-signal evidence.
    base=ROOT/'controller_local_to_return_probe_v1/vector_control'
    old=dict(np.load(base/'motion_raw.npz'));d=dict(np.load(base/'dataset.npz'))
    mapping={(str(s),str(c)):i for i,(s,c) in enumerate(zip(d['state_uid'],d['controller']))}
    ix=np.array([mapping[s,'goal_'+c] for s in sids for c in FAMILIES])
    olo=old['success'][ix][:,[0,5,10,15],:2].reshape(32,4,4,2).astype(float)
    ohi=olo+old['unknown'][ix][:,[0,5,10,15],:2].reshape(32,4,4,2)
    bootstrap=np.random.default_rng(2026101036).integers(32,size=(20000,32))
    weights=np.array([np.bincount(x,minlength=32) for x in bootstrap],float)
    bootout={'indices':bootstrap};rows=[];powers=[]
    a,b=np.triu_indices(4,1)
    complete=~ru.astype(bool).any(-1)
    panels={'state_eta':(rs,complete),'state_controller_eta_pairs':(rs[:,a]-rs[:,b],complete[:,a]&complete[:,b])}
    for name,(y,mask) in panels.items():
        errs=np.stack([y[...,list(kk)].mean(-1)-y.mean(-1) for kk in itertools.combinations(range(4),2)])
        cv,iv=moment_variance(errs,mask,np.ones((1,32)))
        ratio=[];cb=[];ib=[]
        for start in range(0,len(weights),1000):
            x,z=moment_variance(errs,mask,weights[start:start+1000]);cb.append(x);ib.append(z);ratio.append(1-x/np.maximum(z,1e-30))
        reduction=np.concatenate(ratio);bootout[name+'__reduction']=reduction;bootout[name+'__common']=np.concatenate(cb);bootout[name+'__independent']=np.concatenate(ib)
        rows.append(dict(interaction=name,physical_states=32,known_cells=int(mask.sum()),total_cells=int(mask.size),
            common_variance=float(cv[0]),independent_variance=float(iv[0]),
            expected_reduction=float(1-cv[0]/iv[0]) if iv[0]>0 else None,
            conditional_CI95=np.quantile(reduction,[.025,.975]).tolist()))
    for name,axes in (('state_eta',(0,2)),('state_controller_eta',(0,1,2))):
        point=energy(olo,ohi,axes);lower=[];upper=[]
        for start in range(0,len(bootstrap),500):
            jj=bootstrap[start:start+500];_,lo,hi=energy(olo[jj],ohi[jj],tuple(x+1 for x in axes))
            lower.append(lo);upper.append(hi)
        lower=np.concatenate(lower);upper=np.concatenate(upper)
        bootout[name+'__true_energy_lower']=lower;bootout[name+'__true_energy_upper']=upper
        powers.append(dict(interaction=name,independent_original_UID_noise=True,
            point_bounds=[float(point[1]),float(point[2])],
            censor_CI95=[float(np.quantile(lower,.025)),float(np.quantile(upper,.975))]))
    primary=rows[0];signal=powers[0];fraction=float(1-complete.mean())
    passed=primary['expected_reduction'] is not None and primary['expected_reduction']>=.5 and signal['censor_CI95'][0]>0 and fraction<=.05 and not collisions
    result=dict(rows=rows,original_noise_signal=powers,gate_passed=bool(passed),
        numerical_unknown=int(ru.sum()),excluded_train_cells=int((~complete).sum()),excluded_cell_fraction=fraction,
        all_physical_states_retained=True,collisions=collisions,numerical_cases=safety,
        source_rollouts=2048,successes=int(rs.sum()),summed_rollout_seconds=total,
        new_physical_states=0,new_training_fits=0,
        decision='Prepare strictlypairedlong-Qcouplingfits.' if passed else 'DoNOTlaunchQfits/other-scenerollouts: predeclaredlong-Qvariance/powergatefailed.',
        root_cause_closed=False)
    write(HERE/'variance_results.json',result);np.savez_compressed(HERE/'bootstrap.npz',**bootout)
    np.savez_compressed(HERE/'raw_labels.npz',success=rs,unknown=ru,observed=seen,rollout_uid=uids,
        state_uid=np.asarray(sids),controller=np.asarray(FAMILIES),eta=np.asarray(p['eta'])[p['eta_indices']])
    (HERE/'REPORT.txt').write_text('LONG-Q NOISE-COUPLING SOURCE SCREEN\n'+result['decision']+'\n'+
        '\n'.join(str(x) for x in rows+powers)+f'\nnewrollouts=2048;collisions={len(collisions)};numericalunknown={int(ru.sum())};noQfits.\n')
    write(HERE/'completion.json',dict(status='COMPLETE_LONG_Q_VARIANCE_SCREEN',gate_passed=bool(passed),root_cause_closed=False,new_rollout=2048,new_fit=0,report_sha256=sha(HERE/'REPORT.txt')))
    print(dict(gate_passed=bool(passed),variance_rows=rows,signal=powers,numerical_unknown=int(ru.sum()),collisions=len(collisions)))

if __name__=='__main__':run()

