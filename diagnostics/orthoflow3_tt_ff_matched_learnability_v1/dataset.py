"""Matched source-only labels, common train-only normalization, zero TEST reads."""
from __future__ import annotations
import argparse,collections
import numpy as np
from .design import ROOT,MAIN,SCENES,read,write,freeze,sha,canonical,eta_identity,connect
from .cache import protocol
CHAINS=('TT','FF')


def labels_for(db,state,eta,chain,p,n):
    cid=p['controllers'][state['scenario']][chain]['controller_uid'];eid=eta_identity(eta)[0]
    rows={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(state['state_uid'],eid,cid))}
    out=[]
    for k in range(n):
        sk=canonical(dict(future_index=k,future_root=2026100403,rng_namespace=state['rng_namespace']))
        assert sk in rows,('Missing attempted record',state['uid'],chain,k)
        r=rows[sk];assert r['compatibility_quality']=='EXACT_REUSE' and not r['conflict_quarantined']
        out.append(dict(r))
    return out


def materialize():
    assert not (ROOT/'dataset_frozen.json').exists(),'Dataset frozen'
    p=protocol();n=len(p['states']);context=np.full((2,n,16,76),np.nan,np.float32);features=None;seen=[];docs=[]
    assert read(ROOT/'context_smoke.json')['context_sha256']==sha(ROOT/'context.py')
    for w in range(5):
        doc=read(ROOT/'contexts'/f'worker{w}.json');assert doc['context_sha256']==sha(ROOT/'context.py')
        assert doc['protocol_sha256']==sha(ROOT/'protocol.json');docs.append(doc)
        x=np.load(ROOT/'contexts'/f'entities_worker{w}.npz');idx=x['indices'];seen.extend(idx.tolist())
        if features is None:features={k:np.zeros((n,*x[k].shape[1:]),np.float32) for k in x.files if k!='indices'}
        for k in features:features[k][idx]=x[k]
        for r in doc['items']:context[CHAINS.index(r['chain']),r['state_index'],r['eta_index']]=r['context']
    assert sorted(seen)==list(range(n)) and np.isfinite(context).all()
    np.savez_compressed(ROOT/'entities.npz',**features)
    assert min(context[...,73:].mean(axis=(0,1,2)))>.95,'Context validity too low; audit measurement before fitting'
    s=np.full((2,n,16),np.nan,np.float32);f=s.copy();own_s=s.copy();own_f=s.copy();numeric=s.copy();provenance=[]
    with connect(True) as db:
        for i,state in enumerate(p['states']):
            if state['metadata']['split']=='test':continue  # Never open target outcomes here.
            nt=4 if state['metadata']['split']=='train' else 16
            for j,eta in enumerate(p['eta']):
                rr=[labels_for(db,state,eta,chain,p,nt) for chain in CHAINS]
                common=[k for k in range(nt) if not any(rr[ci][k]['numerical_failure'] for ci in range(2))]
                for ci,chain in enumerate(CHAINS):
                    valid=[r for r in rr[ci] if not r['numerical_failure']]
                    s[ci,i,j]=sum(rr[ci][k]['success'] for k in common);f[ci,i,j]=len(common)-s[ci,i,j]
                    own_s[ci,i,j]=sum(r['success'] for r in valid);own_f[ci,i,j]=len(valid)-own_s[ci,i,j]
                    numeric[ci,i,j]=nt-len(valid)
                    provenance.append(dict(state_uid=state['state_uid'],state_index=i,eta_uid=eta_identity(eta)[0],eta_index=j,
                        chain=chain,controller_uid=p['controllers'][state['scenario']][chain]['controller_uid'],
                        split=state['metadata']['split'],all_requested_rollout_uids=[r['rollout_uid'] for r in rr[ci]],
                        common_valid_rollout_uids=[rr[ci][k]['rollout_uid'] for k in common],common_seed_indices=common))
    np.testing.assert_array_equal(s[0]+f[0],s[1]+f[1])
    norms={};cc=context.copy();ee=np.zeros((n,16,3),np.float32);raw_eta=np.array(p['eta'],float)
    for scene in SCENES:
        tr=np.flatnonzero([st['scenario']==scene and st['metadata']['split']=='train' for st in p['states']]);idx=np.flatnonzero([st['scenario']==scene for st in p['states']])
        ec=raw_eta.mean(0);es=np.maximum(raw_eta.std(0),.1);center=np.zeros(76);scale=np.ones(76)
        tt=context[:,tr].reshape(-1,76)
        for a,b,flag,floor in ((0,24,73,.05),(24,40,73,.01),(40,56,74,.05),(56,72,75,.05)):
            v=tt[tt[:,flag]>0,a:b];assert len(v)>0;center[a:b]=v.mean(0);scale[a:b]=np.maximum(v.std(0),floor)
        cc[:,idx]=(context[:,idx]-center)/scale
        for a,b,flag in ((0,40,73),(40,56,74),(56,72,75)):
            # Keep the chain axis first: a scalar advanced index for ``flag``
            # would move the state axis ahead of it in NumPy.
            cc[:,idx,:,a:b]*=(context[:,idx][:,:,:,flag]>0)[...,None]
        ee[idx]=(raw_eta-ec)/es
        norms[scene]=dict(eta_center=ec.tolist(),eta_scale=es.tolist(),context_center=center.tolist(),context_scale=scale.tolist(),
            train_state_indices=tr.tolist(),paired_chains_equal_mass=True,physical='frozen physical-units entity encoder')
    assert np.isfinite(ee).all() and np.isfinite(cc).all()
    np.savez_compressed(ROOT/'dataset.npz',success=s,failure=f,own_success=own_s,own_failure=own_f,numerical=numeric,
        eta=ee,context=cc,raw_context=context,
        scene=np.array([st['scenario'] for st in p['states']]),split=np.array([st['metadata']['split'] for st in p['states']]))
    write(ROOT/'normalization.json',norms);write(ROOT/'canonical_source_keys.json',provenance)
    audit=[]
    for scene in SCENES:
        for split in ('train','validation'):
            ix=np.flatnonzero([st['scenario']==scene and st['metadata']['split']==split for st in p['states']])
            for ci,chain in enumerate(CHAINS):
                audit.append(dict(scene=scene,split=split,chain=chain,states=len(ix),pairs=len(ix)*16,
                    observed_common_trials=int((s+f)[ci,ix].sum()),own_observed_trials=int((own_s+own_f)[ci,ix].sum()),
                    numerical=int(numeric[ci,ix].sum()),zero_information_pairs=int(((s+f)[ci,ix]==0).sum()),
                    common_success_count=int(s[ci,ix].sum()),common_failure_count=int(f[ci,ix].sum())))
    freeze(ROOT/'dataset_frozen.json',dict(dataset_sha256=sha(ROOT/'dataset.npz'),entities_sha256=sha(ROOT/'entities.npz'),
        normalization_sha256=sha(ROOT/'normalization.json'),source_keys_sha256=sha(ROOT/'canonical_source_keys.json'),
        protocol_sha256=sha(ROOT/'protocol.json'),analysis_protocol_sha256=sha(ROOT/'analysis_protocol.json'),
        context_code_sha256=sha(ROOT/'context.py'),model_code_sha256=sha(MAIN/'diagnostics/orthoflow3_critic_rootcause_resolution_v1/db_transfer_train.py'),
        representation_sha256=docs[0]['representation_sha256'],source_family_overlap=0,test_outcomes_read=False,
        common_valid_counts_identical=True,audit=audit,context_errors=[e for doc in docs for e in doc['errors']]))
    print(dict(dataset_frozen=True,audit=audit),flush=True)


def truth():
    assert (ROOT/'models_frozen.json').exists();p=protocol();indices=[i for i,s in enumerate(p['states']) if s['metadata']['split']=='test']
    result=np.zeros((2,len(indices),16,16,3),np.int8);keys=[]
    with connect(True) as db:
        for loc,i in enumerate(indices):
            state=p['states'][i]
            for j,eta in enumerate(p['eta']):
                for ci,chain in enumerate(CHAINS):
                    rr=labels_for(db,state,eta,chain,p,16)
                    result[ci,loc,j]=[[r['success'],r['numerical_failure'],r['collision']] for r in rr]
                    keys.append(dict(state_index=i,eta_index=j,chain=chain,rollout_uids=[r['rollout_uid'] for r in rr]))
    np.savez_compressed(ROOT/'test_truth.npz',indices=np.array(indices),outcomes=result)
    write(ROOT/'canonical_test_keys.json',keys)
    freeze(ROOT/'test_truth_frozen.json',dict(truth_sha256=sha(ROOT/'test_truth.npz'),model_freeze_sha256=sha(ROOT/'models_frozen.json'),
        prediction_freeze_sha256=sha(ROOT/'test_predictions_frozen.json'),states=len(indices),new_task_rollouts=0))
    print(dict(test_truth_materialized=True,states=len(indices),numerical=int(result[...,1].sum()),collisions=int(result[...,2].sum())))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('materialize','truth'));globals()[a.parse_args().action]()
