"""Identical frozen training machinery; change only controller information."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import jax.numpy as jnp
from diagnostics.controller_q_diversity_v1 import train
from diagnostics.controller_q_diversity_v1.train import compact_field
from diagnostics.controller_q_diversity_v1.contact_features import load as old_contact
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from shared_rollout_db.src.rollout_db import connect
from .study import HERE,OLD,read,write,freeze,sha,FAMILIES

FOLDS={'joint':None,'toy':'toy_give_way','ring':'ring_exchange'}
CONDITIONS=('eta_only','alias','operator')

def prepare():
    p=read(HERE/'protocol.json');states={s['state_uid']:s for s in p['states']};groups=defaultdict(list)
    for r in read(HERE/'cases.json'):groups[r['state_uid'],r['controller_name']].append(r)
    original=np.load(OLD/'data/source_augmented128/dataset.npz')
    data={k:original[k].copy() for k in original.files}
    data['feature_alias']=np.pad(np.concatenate([data['nominal'],old_contact(original)],-1),((0,0),(0,0),(0,32)))
    data['feature_operator']=np.asarray(compact_field(jnp.asarray(np.concatenate([data['nominal'],data['field']],-1))))
    new={k:[] for k in data if k!='eta'}
    with connect(True) as db:
        for (sid,name),cases in sorted(groups.items()):
            state=states[sid];f=np.load(HERE/'features'/sid/(name+'.npz'))
            counts=np.zeros((16,3),np.int32);ys=np.zeros((16,16),np.int8);us=ys.copy();cs=ys.copy()
            for c in cases:
                row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row is not None,c['rollout_uid']
                assert not row['conflict_quarantined']
                j,k=c['eta_index'],c['seed'];u=row['numerical_failure'];y=row['success']
                counts[j]+=[y,int(not(y or u)),u];ys[j,k]=y;us[j,k]=u;cs[j,k]=row['collision']
            ent={k[7:]:f[k] for k in f.files if k.startswith('entity_')}
            template={k[7:]:data[k][0] for k in data if k.startswith('entity_')}
            ent={k:v[1] for k,v in rep.batch([template,ent]).items()}
            for k,v in ent.items():new['entity_'+k].append(v)
            n=len(f['nominal']);pad=lambda a:np.pad(a,((0,4-n),(0,0)))
            nominal=pad(f['nominal']);field=pad(f['field']);contact=pad(f['contact'])
            new['nominal'].append(nominal);new['field'].append(field)
            new['feature_alias'].append(np.pad(np.concatenate([nominal,contact],-1),((0,0),(0,32))))
            new['feature_operator'].append(np.asarray(compact_field(jnp.asarray(np.concatenate([nominal,field],-1)))))
            values=dict(success=counts[:,0],failure=counts[:,1],unknown=counts[:,2],seed_success=ys,seed_unknown=us,seed_collision=cs,
                state_uid=sid,controller=name,scene=state['scenario'],split=state['study_split'],state_index=state['study_index'],trials=2 if state['study_split']=='train' else 4)
            for k,v in values.items():new[k].append(v)
    for k in new:data[k]=np.concatenate([data[k],np.asarray(new[k])],axis=0)
    data['original_split']=data['split'].copy();data['split']=data['split'].astype('<U24')
    common=HERE/'combined.npz'
    if not common.exists():np.savez_compressed(common,**data)
    for fold,scene in FOLDS.items():
        out={k:v.copy() for k,v in data.items()}
        if scene is not None:out['split'][out['scene']!=scene]='development_target'
        for condition in CONDITIONS:
            root=HERE/'models'/fold/condition;folder=root/'data/source128';folder.mkdir(parents=True,exist_ok=True)
            path=folder/'dataset.npz'
            if not path.exists():np.savez_compressed(path,**out)
            freeze(folder/'manifest.json',dict(dataset_sha256=sha(path),final_labels_used=False,
                parent_sha256=sha(common),scope='DEVELOPMENT; source-only normalization and checkpoint choice in each scene fold'))
            plan=read(OLD/'training_plan.json');plan['batch_per_scene']=60 if scene is None else 120
            plan['root_condition']=condition;plan['root_checkpoint_rule']='Equal new-family and old-held-controller source-VAL Q+0.5controller contrast error; selection regret/NLL ties.'
            freeze(root/'training_plan.json',plan)
    jobs=[dict(fold=f,condition=c,seed=s) for f in FOLDS for c in CONDITIONS for s in train.SEEDS]
    freeze(HERE/'fit_jobs.json',jobs)
    write(HERE/'dataset_audit.json',dict(rows=len(data['scene']),training_states=len(set(data['state_uid'][data['split']=='train'])),
        old_label_outcomes=int(np.sum(original['success']+original['failure']+original['unknown'])),
        new_family_attempts=sum(len(v) for v in groups.values()),labels_same_for_all_methods=True,
        additional_outcomes_generated=3200,source_compatible_previous_family_outcomes_reused=640))

CONDITION=None
def normalization(d,kind=None):
    key='feature_alias' if CONDITION in ('alias','eta_only') else 'feature_operator'
    good=d['split']=='train';mask=d['entity_agent_mask'].astype(bool);x=d[key][good][mask[good]]
    return dict(feature=key,mean=x.mean(0).tolist(),scale=np.maximum(x.std(0),.05).tolist(),clip=10.,fit_split='train')

def inputs(d,norm):
    c=np.clip((d[norm['feature']]-np.asarray(norm['mean']))/np.asarray(norm['scale']),-10,10)
    c*=d['entity_agent_mask'][...,None]
    x={k[7:]:jnp.asarray(d[k],jnp.float32) for k in d.files if k.startswith('entity_')}
    return x,jnp.asarray(c,jnp.float32),jnp.asarray(d['eta'],jnp.float32)

def metrics(z,d,ix):
    q=1/(1+np.exp(-np.clip(z,-80,80)));s=d['success'][ix];f=d['failure'][ix];u=d['unknown'][ix];n=s+f;good=n>0
    y=s/np.maximum(n,1);mse=float(np.sum((q-y)**2*good)/good.sum())
    nll=float(np.sum(s*np.logaddexp(0,-z)+f*np.logaddexp(0,z))/n.sum())
    sid=d['state_uid'][ix];ctl=d['controller'][ix];scores={};contrasts={}
    for label,mask in [('operator_family',np.isin(ctl,FAMILIES)),('old_held_controller',np.isin(ctl,['analytic4','analytic5']))]:
        values=[]
        for state in np.unique(sid[mask]):
            ids=np.flatnonzero((sid==state)&mask);aa,bb=np.triu_indices(len(ids),1);aa=ids[aa];bb=ids[bb]
            valid=good[aa]&good[bb];values.extend((((q[aa]-q[bb])-(y[aa]-y[bb]))**2)[valid].tolist())
        cm=float(np.mean(values)) if values else 0.
        pm=float(np.mean(((q-y)**2)[good&mask[:,None]]))
        scores[label]=pm+.5*cm;contrasts[label]=cm
    eligible=s+f+u>0;lo=s/np.maximum(s+f+u,1);selected=np.argmax(np.where(eligible,q,-np.inf),axis=1)
    reg=float(np.mean(np.max(np.where(eligible,lo,-np.inf),axis=1)-lo[np.arange(len(ix)),selected]))
    return dict(score=float(np.mean(list(scores.values()))),probability_mse=mse,nll=nll,selection_regret=reg,
                controller_contrast_mse=float(np.mean(list(contrasts.values()))),**{k+'_score':v for k,v in scores.items()})

def fit(index):
    global CONDITION
    job=read(HERE/'fit_jobs.json')[index];CONDITION=job['condition'];train.HERE=HERE/'models'/job['fold']/CONDITION
    train.normalization=normalization;train.inputs=inputs;train.metrics=metrics
    kind='eta_only' if CONDITION=='eta_only' else 'state_field_compact_bilinear'
    train.fit(kind,job['seed'],variant='source128')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:fit(a.index)
