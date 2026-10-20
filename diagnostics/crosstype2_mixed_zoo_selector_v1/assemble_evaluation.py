from common import *
from baselines import few_label
from statistics_pipeline import best


def main():
    assert (HERE/'new_evaluation_COMPLETE.receipt.json').exists()
    assert (HERE/'locked_few_labels_COMPLETE.receipt.json').exists()
    p=read(HERE/'new_evaluation_protocol.json');names=p['controller_order'];indices={n:i for i,n in enumerate(names)};shape=(len(names),24,17,2)
    labels=np.full(shape,-2,np.int8);uids=np.full(shape,'',dtype='U96');events={k:np.zeros(shape,np.int8) for k in ('collision','timeout','unknown')};traces=[]
    with rollout_db.connect(True) as db:
        for c in read(HERE/'new_evaluation_CASES.json'):
            i,s,a,q=indices[c['base_controller_name']],c['phase_state_index'],c['candidate_index'],c['seed']
            row=db.execute('SELECT success,numerical_failure,collision,timeout FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row is not None
            labels[i,s,a,q]=-1 if row['numerical_failure'] else row['success'];uids[i,s,a,q]=c['rollout_uid']
            for key in events:events[key][i,s,a,q]=row['numerical_failure'] if key=='unknown' else row[key]
            raw=HERE/'new_evaluation/raw'/f"{c['rollout_uid']}.json"
            if raw.exists():
                r=read(raw);assert sha(r['trace_path'])==r['trace_sha256'];traces.append(dict(controller=c['base_controller_name'],state_index=s,candidate_index=a,seed=q,rollout_uid=c['rollout_uid'],trace_path=r['trace_path'],trace_sha256=r['trace_sha256'],outcome=r['outcome']))
        locked_names=read(HERE/'LOCKED_CAPABILITY.json')['eligible'];lookup={n:i for i,n in enumerate(locked_names)};few=np.full((len(locked_names),8,16),-2,np.int8)
        for c in read(HERE/'locked_few_labels_CASES.json'):
            row=db.execute('SELECT success,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row is not None
            few[lookup[c['base_controller_name']],c['state_index'],c['candidate_index']]=-1 if row['numerical_failure'] else row['success']
    assert not np.any(labels==-2) and not np.any(few==-2)
    with np.load(HERE/'NEW_EVAL_CHOICES.npz',allow_pickle=False) as z:methods={k:z[k].copy() for k in z.files if k not in ('names','types','state_uids')}
    for k in (2,4,8):
        chosen=best((few[:,:k]==1).mean(1))
        for name,a in zip(locked_names,chosen):methods[f'few_{k}'][indices[name]]=a
    assert all(np.all((v>=0)&(v<17)) for v in methods.values())
    np.savez_compressed(HERE/'NEW_EVAL_BUNDLE.npz',names=names,types=[p['types'][n] for n in names],labels=labels,uids=uids,state_uids=[s['state_uid'] for s in p['states']],**events)
    np.savez_compressed(HERE/'NEW_EVAL_ANALYSIS_CHOICES.npz',names=names,types=[p['types'][n] for n in names],**methods)
    write(HERE/'NEW_EVAL_TRACE_INDEX.json',traces)
    seal('NEW_EVAL_BUNDLE.receipt.json',[HERE/'NEW_EVAL_BUNDLE.npz',HERE/'NEW_EVAL_ANALYSIS_CHOICES.npz',HERE/'NEW_EVAL_TRACE_INDEX.json',HERE/'new_evaluation_COMPLETE.receipt.json',HERE/'locked_few_labels_COMPLETE.receipt.json',HERE/'assemble_evaluation.py'],frozen_primary_choices_unchanged=True,locked_few_labels_only_after_confirmation=True,unknowns_preserved=True)
    print('Full new-state bundle and post-confirmation few-label baselines assembled')


if __name__=='__main__':main()
