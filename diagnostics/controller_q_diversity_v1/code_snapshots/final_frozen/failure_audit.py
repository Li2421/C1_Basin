"""Trace every confirmation failure to exact cached seed evidence."""
import csv
import json
from collections import Counter,defaultdict
import numpy as np
from .experiment import HERE,read,write,connect,sha
from .evaluate import csvwrite


def run():
    assert (HERE/'evaluation_complete.json').exists()
    primary=read(HERE/'models_frozen.json')['primary'];pred=np.load(HERE/'confirmation_predictions.npz')
    d=np.load(HERE/'data/final/dataset.npz');q=pred[primary].mean(0)
    indexes={(str(s),str(c)):i for i,(s,c) in enumerate(zip(d['state_uid'],d['controller']))}
    picks={(s,c):int(q[i].argmax()) for (s,c),i in indexes.items()}
    counters=defaultdict(Counter);failure_rows=[];collision_rows=[];numerical_messages=Counter()
    attempts=[]
    for stage in ('confirmation_budget','safety_confirmation'):
        attempts.extend(read(HERE/'batches'/stage/'cases.json'))
    with connect(True) as db:
        for case in attempts:
            r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(case['rollout_uid'],)).fetchone()
            assert r is not None and not r['conflict_quarantined']
            name=case['controller_name'];is_safety=name.endswith('_safety');ctl=name.removesuffix('_safety')
            counts=counters[(case['scene'],name)];counts.update(attempts=1)
            for field in ('success','collision','deadlock','timeout','numerical_failure'):counts[field]+=r[field]
            counts['episode_steps']+=r['episode_length']
            selected=is_safety or case['eta_index']==picks[(case['state_uid'],ctl)]
            if selected:
                for field in ('success','collision','deadlock','timeout','numerical_failure'):counts['selected_'+field]+=r[field]
                counts['selected_attempts']+=1
            if r['success'] and not r['collision']:continue
            record=dict(scene=case['scene'],state_uid=case['state_uid'],controller=name,
                eta_index=case['eta_index'],seed=case['seed'],rollout_uid=case['rollout_uid'],
                outcome=r['outcome'],steps=r['episode_length'],selected_by_primary_or_safety=selected,
                success=r['success'],collision=r['collision'],deadlock=r['deadlock'],timeout=r['timeout'],
                numerical_failure=r['numerical_failure'],original_source_file=r['original_source_file'])
            raw=HERE/'batches/confirmation_joint/raw'/(case['rollout_uid']+'.json')
            if raw.exists():
                details=read(raw);record['raw_record']=str(raw)
                error=details.get('error') or {};record['error_type']=error.get('type');record['error_message']=error.get('message')
                if r['numerical_failure']:numerical_messages[(case['scene'],str(error.get('type')),str(error.get('message')))]+=1
                if r['collision']:
                    record['simulation_summary']=json.dumps(details['summary'],sort_keys=True)
            if r['collision']:collision_rows.append(record.copy())
            failure_rows.append(record)
    csvwrite(HERE/'failure_cases.csv',failure_rows)
    if collision_rows:csvwrite(HERE/'collision_cases.csv',collision_rows)
    predictions=[];unknown=d['seed_unknown'].sum(-1);known=16-unknown
    truth=d['seed_success'].sum(-1)/np.maximum(known,1)
    for (sid,ctl),i in indexes.items():
        valid=known[i]>0
        mse=float(np.mean((q[i]-truth[i])[valid]**2)) if valid.any() else None
        j=int(q[i].argmax());oracle=int(d['seed_success'][i,:,8:].mean(-1).argmax())
        predictions.append(dict(scene=str(d['scene'][i]),state_uid=sid,controller=ctl,
            Q_MSE_known=mse,known_candidates=int(valid.sum()),eta_selected=j,eta_eval_oracle=oracle,
            predicted_selected_Q=float(q[i,j]),observed_selected_Q=float(truth[i,j]),
            selected_unknown_seeds=int(unknown[i,j]),
            eval_regret=float(d['seed_success'][i,oracle,8:].mean()-d['seed_success'][i,j,8:].mean())))
    csvwrite(HERE/'per_state_Q_errors.csv',predictions)
    write(HERE/'failure_audit.json',dict(primary=primary,total_requests=len(attempts),
        counts=[dict(scene=k[0],controller=k[1],**v) for k,v in counters.items()],
        numerical_messages=[dict(scene=k[0],exception=k[1],message=k[2],count=v) for k,v in numerical_messages.items()],
        collision_count=len(collision_rows),all_failures_exported=len(failure_rows),
        collision_review_required=bool(collision_rows),numerical_policy='Unresolved solver outcomes; never relabel as physical failure or rerun to replace evidence',
        code_sha256=sha(__file__)))
    print('Failure evidence exported',len(failure_rows),'collisions',len(collision_rows),flush=True)


if __name__=='__main__':run()
