"""Align a diagnostic critic with actual verified runtime success.

Numerical exceptions remain unresolved physical outcomes in the authoritative
database. For this explicitly different operational estimand, an exception did
not deliver success. No physics labels, main models, or evaluator are replaced.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from . import train
from .experiment import HERE,read,write,freeze,sha
from .evaluate import donor_bootstrap,interval,csvwrite

OUT=HERE/'checks/operational_target'
MODELS=('eta_only','state_only','state_contact_bilinear')
VARIANT='source_augmented128'


def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.is_symlink():dst.symlink_to(src,target_is_directory=src.is_dir())


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('training_plan.json','protocol.json','confirmation_allocation.json','features'):
        link(HERE/name,OUT/name)
    d=np.load(HERE/'data'/VARIANT/'dataset.npz');data={k:d[k] for k in d.files}
    data['physical_numerical_unknown']=data['unknown'].copy()
    data['failure']=data['failure']+data['unknown'];data['unknown']=np.zeros_like(data['unknown'])
    folder=OUT/'data'/VARIANT;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'dataset.npz';assert not path.exists();np.savez_compressed(path,**data)
    freeze(folder/'manifest.json',dict(dataset_sha256=sha(path),parent_sha256=sha(HERE/'data'/VARIANT/'dataset.npz'),
        final_labels_used=False,target='Verified runtime success, including numerical non-delivery in denominator',
        authoritative_physical_outcomes_changed=False,new_rollouts=0))
    freeze(OUT/'diagnostic_protocol.json',dict(code_sha256=sha(__file__),final_outcomes_inspected=False,
        primary='state_contact_bilinear',models=list(MODELS),seeds=list(train.SEEDS),
        purpose='Test target mismatch: censored physical-success likelihood versus actual runtime success',
        final_scope='Supplemental pre-outcome control; same TRAIN attempts, inputs, optimizer, source-only checkpoint rule; numerical events are not called physical collisions/failures',
        new_rollouts=0))


def fit(index):
    train.HERE=OUT;kind=MODELS[index//3];seed=train.SEEDS[index%3]
    train.fit(kind,seed,variant=VARIANT)


def freeze_predictions():
    assert not (HERE/'evaluation_complete.json').exists()
    records=[]
    for kind in MODELS:
        for seed in train.SEEDS:
            path=OUT/'fits'/VARIANT/kind/f'seed{seed}';r=read(path/'complete.json')
            r.update(path=str(path));records.append(r)
    freeze(OUT/'models_frozen.json',dict(records=records,primary='state_contact_bilinear',data_variant=VARIANT,train_limit=128,
        target='Verified operational success, not hypothetical physical outcome after numerical exceptions'))
    from . import predict
    predict.HERE=OUT;predict.KINDS=MODELS;predict.run()
    print('Operational-target diagnostic predictions frozen before final labels were opened')


def evaluate():
    assert (HERE/'evaluation_complete.json').exists()
    d=np.load(HERE/'data/final/dataset.npz');p=np.load(OUT/'confirmation_predictions.npz')
    old=np.load(HERE/'confirmation_predictions.npz');primary='state_contact_bilinear'
    for key in ('state_uid','scene','controller'):np.testing.assert_array_equal(d[key],p[key])
    tables=[];comparisons=[];perstate=[];samples={}
    safety={(r['scene'],r['state_uid'],r['controller']):r for r in csv.DictReader(open(HERE/'per_state_decisions.csv')) if r['method']=='Safety'}
    for scene in sorted(set(d['scene'])):
        ids=np.flatnonzero(d['scene']==scene);n=len(ids)//2
        ss=d['seed_success'][ids].reshape(n,2,16,16);uu=d['seed_unknown'][ids].reshape(n,2,16,16)
        target=ss.mean(-1);ev=ss[...,8:].mean(-1);robust=(ss.sum(-1)>=15).astype(float)
        pred={k:p[k][:,ids].mean(0).reshape(n,2,16) for k in MODELS}
        pred['conditional_target_original']=old[primary][:,ids].mean(0).reshape(n,2,16)
        pred['controller_mismatch']=p[primary+'_controller_mismatch'][:,ids].mean(0).reshape(n,2,16)
        row=np.arange(n)[:,None];ctl=np.arange(2)[None,:];stats={}
        boot=np.random.default_rng(2026100955).integers(n,size=(20000,n))
        for name,q in pred.items():
            choice=q.argmax(-1);selected=ev[row,ctl,choice];b15=robust[row,ctl,choice]
            mse=((q-target)**2).mean((1,2));stats[name]=dict(mse=mse,eval_success=selected.mean(1),B15=b15.mean(1))
            for ci in range(2):
                tables.append(dict(scene=str(scene),controller=f'analytic{ci+6}',method=name,physical_states=n,
                    verified_Q_MSE=float(((q[:,ci]-target[:,ci])**2).mean()),B15_count=int(b15[:,ci].sum()),
                    evaluation8_success=float(selected[:,ci].mean()),selected_numerical=int(uu.sum(-1)[row,ctl,choice][:,ci].sum())))
            for si in range(n):
                for ci in range(2):perstate.append(dict(scene=str(scene),state_uid=str(d['state_uid'][ids[si*2+ci]]),controller=f'analytic{ci+6}',method=name,
                    eta_index=int(choice[si,ci]),evaluation8_success=float(selected[si,ci]),B15=float(b15[si,ci]),verified_Q_MSE=float(((q[si,ci]-target[si,ci])**2).mean())))
        safe=[safety[(str(scene),str(d['state_uid'][i]),str(d['controller'][i]))] for i in ids]
        stats['Safety']=dict(eval_success=np.asarray([float(v['evaluation_success']) for v in safe]).reshape(n,2).mean(1),
            B15=np.asarray([float(v['B15']) for v in safe]).reshape(n,2).mean(1))
        def record(label,estimate,draws):
            comparisons.append(dict(scene=str(scene),comparison=label,physical_states=n,mean=float(estimate),CI95=interval(draws)))
            samples[str(scene)+'_'+label]=draws
        for name in stats:
            if name==primary:continue
            for metric in stats[name]:
                value=(stats[name][metric]-stats[primary][metric]) if metric=='mse' else (stats[primary][metric]-stats[name][metric])
                record(name+'_'+metric,float(value.mean()),value[boot].mean(1))
        choice=pred[primary].argmax(-1)
        for metric,truth in [('eval_success',ev),('B15',robust)]:
            matrix=np.mean(np.stack([truth[:,ci,choice[:,ci]] for ci in range(2)],-1),-1)
            estimate,bs,valid=donor_bootstrap(matrix,boot);observed=stats[primary][metric]
            record('matched-minus-exact-state-mismatch_'+metric,observed.mean()-estimate,observed[boot].mean(1)[valid]-bs[valid])
            fixed=truth[boot].mean(1).max(-1).mean(-1)
            record('matched-minus-bootstrap-best-fixed_'+metric,observed.mean()-truth.mean(0).max(-1).mean(),observed[boot].mean(1)-fixed)
        q=pred[primary];matrix=((q[None,:,:,:]-target[:,None,:,:])**2).mean((2,3))
        estimate,bs,valid=donor_bootstrap(matrix,boot)
        record('exact-state-mismatch-minus-matched_Q_MSE',estimate-stats[primary]['mse'].mean(),bs[valid]-stats[primary]['mse'][boot].mean(1)[valid])
    csvwrite(OUT/'selection_tables.csv',tables);csvwrite(OUT/'per_state.csv',perstate)
    write(OUT/'paired_comparisons.json',comparisons);np.savez_compressed(OUT/'bootstrap.npz',**samples)
    write(OUT/'evaluation_complete.json',dict(target='Observed verified operational success; numerical non-delivery counts against operational success only',
        physical_failure_relabeling=False,main_models_unchanged=True,new_rollouts=0,primary=primary,
        limitations='Supplemental target-alignment control; do not replace physical-success inference or its censoring bounds with this estimand. State bootstrap conditional on two held controllers.'))
    print('Operational-target confirmation control complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','freeze','evaluate']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='fit':fit(a.index)
    elif a.action=='freeze':freeze_predictions()
    else:evaluate()
