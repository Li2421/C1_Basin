"""Frozen zero-new-rollout source data/context augmentation experiment."""
import argparse
import collections
import copy
import hashlib
import os
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from .db_transfer_data import ROOT, OUT as OLD, SCENES, read, write, sha
from .source_joint_support_audit import OUT, family
from .source_controller_cv_v2 import dependencies

ARMS=('matched_repeat','true_t0_base','crossed_controller')


def prepare():
    assert not (OUT/'protocol.json').exists(), 'Already frozen'
    audit=read(OUT/'cache_audit.json');assert not audit['excluded']
    rows=read(OLD/'pairs.json');states=read(OLD/'states.json');ctl=read(OLD/'controllers.json')
    oldcount=len(rows)
    for r in rows:r['augmentation']=False
    extra=read(OUT/'cached_source_rows.json');st=read(OUT/'cached_source_states.json')
    cc=read(OUT/'cached_source_controllers.json');ctl.update(cc)
    existing={s['state_uid']:i for i,s in enumerate(states)}
    for s in st:
        if s['state_uid'] not in existing:
            existing[s['state_uid']]=len(states);states.append(s)
    for r in extra:
        assert not r['existing_pair']
        r.update(state_index=existing[r['state_uid']],augmentation=True)
    rows+=extra
    profiles=read(ROOT.parents[1]/'diagnostics/orthoflow3_controller_training_repair_v1/protocol.json')['profiles']
    base=profiles[0]['controller_uid'];alt=profiles[1]['controller_uid'];held=profiles[2]['controller_uid']
    heldhash=profiles[2]['sha256']
    excluded={uid for uid,profile in ctl.items() if heldhash in dependencies(profile)}
    fit=np.array([i for i,r in enumerate(rows[:oldcount]) if r['split']=='train' and r['controller_uid'] not in excluded])
    seen=np.array([i for i,r in enumerate(rows[:oldcount]) if r['split']=='validation' and r['controller_uid'] not in excluded])
    augbase=np.array([i for i,r in enumerate(rows) if r['augmentation'] and r['split']=='train' and r['controller_uid']==base])
    augcross=np.array([i for i,r in enumerate(rows) if r['augmentation'] and r['split']=='train' and r['controller_uid'] in (base,alt)])
    augheld=np.array([i for i,r in enumerate(rows) if r['augmentation'] and r['split']=='validation' and r['controller_uid']==held])
    augseen=np.array([i for i,r in enumerate(rows) if r['augmentation'] and r['split']=='validation' and r['controller_uid'] in (base,alt)])
    source_ring=np.array([i for i in fit if rows[i]['scenario']=='ring_exchange' and rows[i]['pool']=='intervention'])
    fitids=np.r_[fit,augcross]
    valids=np.r_[seen,augheld,augseen]
    tf={(rows[i]['scenario'],family(rows[i])) for i in fitids};vf={(rows[i]['scenario'],family(rows[i])) for i in valids}
    assert not tf&vf
    assert all(heldhash not in dependencies(ctl[rows[i]['controller_uid']]) for i in fitids)
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
    xx=rep.batch([rep.entities(s['physical']) for s in states])
    # Shape change is allowed only as mask padding; native source entities must
    # remain identical in their original coordinates.
    prior=np.load(OLD/'source_entities.npz')
    for k in prior.files:
        old=prior[k];slices=tuple(slice(0,n) for n in old.shape)
        np.testing.assert_array_equal(xx[k][slices],old)
    np.savez_compressed(OUT/'entities.npz',**xx)
    np.savez_compressed(OUT/'indices.npz',fit=fit,seen=seen,augheld=augheld,augseen=augseen,
        matched_repeat=source_ring,true_t0_base=augbase,crossed_controller=augcross)
    write(OUT/'pairs.json',rows);write(OUT/'states.json',states);write(OUT/'controllers.json',ctl)
    write(OUT/'protocol.json',dict(
      hypothesis='Separate omitted true-t0 wide-eta exposure from crossed future-controller supervision using compatible cached outcomes only.',
      arms=ARMS,models=['eta_only','full_context','additive_nominal_context_crossed_only'],seeds=[17,23,41],
      old_pairs=oldcount,added_pairs=len(extra),cached_valid_trials=audit['observed_continuations'],
      held_source_controller=held,held_source_future_flow_sha256=heldhash,excluded_controller_uids=sorted(excluded),
      fit_source_pairs=len(fit),augmentation_train_pairs={'matched_repeat':len(source_ring),'true_t0_base':len(augbase),'crossed_controller':len(augcross)},
      validation=dict(seen=len(seen),held_controller_wide=len(augheld),seen_controller_wide=len(augseen),
         selection='Equal mean of (1) six scene/pool mean continuation NLL on original source VAL, (2) held v7 controller wide-eta VAL NLL. Same common step per arm/model selected by mean over three seeds. Source validation only, not independent confirmation.'),
      sampler='Six scene/pool groups,48 rows each. In Ring intervention group draw24 old+24 arm-specific; matched-repeat uses old labels twice. Fixed group-mean observed-trial normalization, no ranking loss.',
      normalization='Shared across all arms; old fitting rows only, held v7/parents excluded; no added rows or VAL statistics. Isolates supervision rather than normalization.',
      training='Same architecture,initialization,uniform random draws,AdamW8e-4 wd1e-4 clip5;2000steps;eval100;raw skip frozen first25steps',
      context='Unchanged76D portable physical response; derived measurements only, no success continuations',
      target_use='No target labels, scores, thresholds or checkpoint choice. Previously opened88138/88139 can only be posthoc regression after source freeze. Any new independent claim requires a new predeclared controller/family confirmation.',
      generator_changed=False,new_rollouts=0,source_family_overlap=0,
      input_code_sha256=sha(__file__),model_code_sha256=sha(ROOT/'db_transfer_train.py'),
      context_code_sha256=sha(ROOT/'db_transfer_context.py'),old_data_sha256=sha(OLD/'pairs.json'),
      pairs_sha256=sha(OUT/'pairs.json')))
    print(dict(prepared=True,old_pairs=oldcount,added=len(extra),held_val=len(augheld),new_rollouts=0),flush=True)


def features(worker,workers=5):
    from .db_transfer_context import Measure
    protocol=read(OUT/'protocol.json');assert protocol['input_code_sha256']==sha(__file__)
    assert protocol['context_code_sha256']==sha(ROOT/'db_transfer_context.py')
    rows=read(OUT/'pairs.json');states=read(OUT/'states.json');ctl=read(OUT/'controllers.json')
    chosen=[i for i in range(protocol['old_pairs'],len(rows)) if (i-protocol['old_pairs'])%workers==worker]
    values=np.zeros((len(chosen),76),np.float32);errors=[];start=time.perf_counter();current=None
    for local,i in sorted(enumerate(chosen),key=lambda t:(rows[t[1]]['controller_uid'],rows[t[1]]['state_uid'])):
        r=rows[i]
        if current!=r['controller_uid']:
            current=r['controller_uid'];p=ctl[current];assert sha(p['path'])==p['sha256']
            measure=Measure(r['scenario'],p['path'])
        values[local],err=measure.one(r['state_uid'],states[r['state_index']]['physical'],np.array(r['eta']))
        errors.extend(dict(index=i,**e) for e in err)
    dest=OUT/'features';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/f'worker{worker}.npz',indices=np.array(chosen),context=values)
    write(dest/f'worker{worker}.json',dict(pairs=len(chosen),errors=errors,seconds=time.perf_counter()-start,
       new_rollouts=0,input_code_sha256=sha(__file__),context_code_sha256=protocol['context_code_sha256']))
    print(dict(worker=worker,pairs=len(chosen),invalid_groups=len(errors),seconds=time.perf_counter()-start),flush=True)


def materialize():
    protocol=read(OUT/'protocol.json');rows=read(OUT/'pairs.json');d=np.load(OUT/'indices.npz')
    raw=np.zeros((len(rows),76),np.float32);raw[:protocol['old_pairs']]=np.load(OLD/'source_context.npz')['context']
    saw=[];errors=[]
    for w in range(5):
        zz=np.load(OUT/'features'/f'worker{w}.npz');doc=read(OUT/'features'/f'worker{w}.json')
        assert doc['input_code_sha256']==sha(__file__) and doc['context_code_sha256']==sha(ROOT/'db_transfer_context.py')
        raw[zz['indices']]=zz['context'];saw.extend(zz['indices'].tolist());errors+=doc['errors']
    assert sorted(saw)==list(range(protocol['old_pairs'],len(rows)))
    assert np.isfinite(raw).all() and (raw[protocol['old_pairs']:,73:76]>0).mean()>.95
    eta=np.array([r['eta'] for r in rows],np.float32);weight=np.zeros(len(rows))
    for sc in SCENES:
        ii=np.array([i for i in d['fit'] if rows[i]['scenario']==sc]);weight[ii]=1/len(ii)/len(SCENES)
    ec=(weight[:,None]*eta).sum(0);es=np.maximum(np.sqrt((weight[:,None]*(eta-ec)**2).sum(0)),.1)
    cc=np.zeros(76);cs=np.ones(76)
    for a,b,flag,floor in ((0,24,73,.05),(24,40,73,.01),(40,56,74,.05),(56,72,75,.05)):
        ww=weight*(raw[:,flag]>0);ww/=ww.sum();cc[a:b]=(ww[:,None]*raw[:,a:b]).sum(0)
        cs[a:b]=np.maximum(np.sqrt((ww[:,None]*(raw[:,a:b]-cc[a:b])**2).sum(0)),floor)
    c=(raw-cc)/cs
    for a,b,flag in ((0,40,73),(40,56,74),(56,72,75)):c[:,a:b]*=raw[:,flag,None]>0
    np.savez_compressed(OUT/'arrays.npz',eta=((eta-ec)/es).astype(np.float32),context=c.astype(np.float32),
        raw_context=raw,state_index=np.array([r['state_index'] for r in rows]),
        success=np.array([r['s'] for r in rows],np.float32),failure=np.array([r['f'] for r in rows],np.float32))
    write(OUT/'normalization.json',dict(eta_center=ec.tolist(),eta_scale=es.tolist(),context_center=cc.tolist(),context_scale=cs.tolist(),
       fitting_rows=d['fit'].tolist(),source_only=True))
    write(OUT/'input_complete.json',dict(pairs=len(rows),invalid_groups=errors,new_rollouts=0,arrays_sha256=sha(OUT/'arrays.npz'),
       entities_sha256=sha(OUT/'entities.npz'),normalization_sha256=sha(OUT/'normalization.json'),protocol_sha256=sha(OUT/'protocol.json')))
    print(dict(materialized=len(rows),errors=len(errors),new_rollouts=0),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','features','materialize']);p.add_argument('--worker',type=int,default=0);a=p.parse_args()
    features(a.worker) if a.action=='features' else globals()[a.action]()
