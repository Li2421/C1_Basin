"""Controlled actualFF four-scene source-only fits; no held-scene selection."""
import argparse
from collections import defaultdict
import numpy as np
import jax.numpy as jnp
from diagnostics.controller_q_diversity_v1 import train
from diagnostics.controller_q_diversity_v1.train import compact_field
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.controller_operator_repair_v1 import learn as two
from shared_rollout_db.src.rollout_db import connect
from .study import HERE,REPAIR,read,write,freeze,sha,FAMILIES

FOLDS={'joint':None,'held_toy':'toy_give_way','held_db':'double_bottleneck','held_four':'four_way_intersection','held_ring':'ring_exchange'}
CONDITIONS=('eta_only','alias','operator')

def prepare():
    p=read(HERE/'protocol.json');states={s['state_uid']:s for s in p['states']};groups=defaultdict(list)
    for r in read(HERE/'cases.json'):groups[r['state_uid'],r['controller_name']].append(r)
    ent=[];rows=[];meta=[];nom=[];field=[]
    with connect(True) as db:
        for (sid,name),cases in sorted(groups.items()):
            state=states[sid];features=np.load(HERE/'features'/sid/(name+'.npz'))
            counts=np.zeros((16,3),np.int32);y=np.zeros((16,16),np.int8);u=y.copy();col=y.copy()
            for c in cases:
                r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is not None,c['rollout_uid']
                assert not r['conflict_quarantined']
                j,k=c['eta_index'],c['seed'];num=r['numerical_failure'];suc=r['success']
                counts[j]+=[suc,int(not(num or suc)),num];y[j,k]=suc;u[j,k]=num;col[j,k]=r['collision']
            rows.append(dict(success=counts[:,0],failure=counts[:,1],unknown=counts[:,2],seed_success=y,seed_unknown=u,seed_collision=col))
            ent.append({k[7:]:features[k] for k in features.files if k.startswith('entity_')})
            pad=lambda a:np.pad(a,((0,4-len(a)),(0,0)))
            nom.append(pad(features['nominal']));field.append(pad(features['field']))
            meta.append(dict(scene=state['scenario'],state_uid=sid,controller=name,split=state['study_split'],original_split=state['study_split'],state_index=state['study_index']))
    entity=rep.batch(ent);data={'entity_'+k:v for k,v in entity.items()}
    data.update({k:np.asarray([r[k] for r in rows]) for k in rows[0]})
    data.update({k:np.asarray([m[k] for m in meta]) for k in meta[0]})
    data.update(eta=np.asarray(p['eta'],np.float32),nominal=np.asarray(nom),field=np.asarray(field))
    data['feature_alias']=np.pad(data['nominal'],((0,0),(0,0),(0,50)))
    data['feature_operator']=np.asarray(compact_field(jnp.asarray(np.concatenate((data['nominal'],data['field']),-1))))
    data['split']=data['split'].astype('<U24')
    assert len(data['scene'])==800
    # Verify exact physical/controller-fingerprint alias, not just rounded
    # descriptor similarity. All five values of Q have the same alias input.
    for sid in np.unique(data['state_uid']):
        ii=np.flatnonzero(data['state_uid']==sid);assert len(ii)==5
        np.testing.assert_array_equal(data['feature_alias'][ii],np.repeat(data['feature_alias'][ii[:1]],5,axis=0))
    if not (HERE/'combined.npz').exists():np.savez_compressed(HERE/'combined.npz',**data)
    for fold,held in FOLDS.items():
        out={k:v.copy() for k,v in data.items()}
        if held is not None:out['split'][out['scene']==held]='development_target'
        for condition in CONDITIONS:
            root=HERE/'models'/fold/condition;folder=root/'data/source128';folder.mkdir(parents=True,exist_ok=True)
            path=folder/'dataset.npz'
            if not path.exists():np.savez_compressed(path,**out)
            freeze(folder/'manifest.json',dict(dataset_sha256=sha(path),final_labels_used=False,held_scene=held,
                actual_TRAIN_states_per_source_scene=32,actual_VAL_states_per_source_scene=8,
                info='source128 is only a legacy trainer directory alias; actual counts are recorded here. No held labels in fit/normalization/selection.'))
            plan=read(REPAIR/'models/joint/operator/training_plan.json')
            plan['batch_per_scene']=30 if held is None else 40
            plan['root_condition']=condition
            plan['root_checkpoint_rule']='Source-validation probability MSE+0.5controller contrast MSE; selection regret and NLL tie break.'
            freeze(root/'training_plan.json',plan)
    jobs=[dict(fold=f,condition=c,seed=s) for f in FOLDS for c in CONDITIONS for s in train.SEEDS]
    freeze(HERE/'fit_jobs.json',jobs)
    write(HERE/'dataset_audit.json',dict(states=160,TRAIN_states_per_scene=32,VAL_states_per_scene=8,
        rows=800,requested_outcomes=7680,alias_exact_for_all_states=True,
        representation='existing physical-only masked entities; nominal14 padded64 or existing compact static-field64; no scene or controller ID'))

def metrics(z,d,ix):
    q=1/(1+np.exp(-np.clip(z,-80,80)));s=d['success'][ix];f=d['failure'][ix];u=d['unknown'][ix];good=s+f>0;y=s/np.maximum(s+f,1)
    mse=float(np.mean(((q-y)**2)[good]));contrasts=[];pair_correct=[]
    for sid in np.unique(d['state_uid'][ix]):
        ii=np.flatnonzero(d['state_uid'][ix]==sid);aa,bb=np.triu_indices(len(ii),1);aa=ii[aa];bb=ii[bb];valid=good[aa]&good[bb]
        contrasts.extend((((q[aa]-q[bb])-(y[aa]-y[bb]))**2)[valid].tolist())
    cmse=float(np.mean(contrasts));eligible=s+f+u>0;lo=s/np.maximum(s+f+u,1)
    chosen=np.argmax(np.where(eligible,q,-np.inf),1)
    regret=float(np.mean(np.max(np.where(eligible,lo,-np.inf),1)-lo[np.arange(len(ix)),chosen]))
    aa,bb=np.triu_indices(16,1);strong=(abs(y[:,aa]-y[:,bb])>=.5)&good[:,aa]&good[:,bb]
    pa=np.sign(q[:,aa]-q[:,bb]);direction=np.sign(y[:,aa]-y[:,bb])
    accuracy=float(np.mean(((pa==direction)+.5*(pa==0))[strong])) if strong.any() else None
    return dict(score=mse+.5*cmse,probability_mse=mse,controller_contrast_mse=cmse,selection_regret=regret,pairwise_accuracy=accuracy,
        nll=float(np.sum(s*np.logaddexp(0,-z)+f*np.logaddexp(0,z))/np.sum(s+f)))

def fit(index):
    job=read(HERE/'fit_jobs.json')[index];two.CONDITION=job['condition']
    train.HERE=HERE/'models'/job['fold']/job['condition'];train.normalization=two.normalization;train.inputs=two.inputs;train.metrics=metrics
    kind='eta_only' if job['condition']=='eta_only' else 'state_field_compact_bilinear'
    train.fit(kind,job['seed'],limit=32,variant='source128')

def review():
    from diagnostics.controller_operator_repair_v1 import review as parent
    parent.HERE=HERE;parent.FOLDS=FOLDS;parent.CONDITIONS=CONDITIONS;parent.run()
    report=read(HERE/'development_results.json')
    for r in report['rows']:
        held=FOLDS[r['fold']]
        r['panel']='held_scene_development' if held==r['scene'] else 'source_validation'
    report['folds']=FOLDS;report['scope']='ActualFF four-scene mechanism diagnosis, not historical terminal-operator LOSO; all target results are DEVELOPMENT.'
    write(HERE/'development_results.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','review']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.action=='fit':fit(a.index)
    else:globals()[a.action]()
