"""Read complete final-candidate labels from the authoritative shared DB."""
from common import *


def main():
    phase='new_evaluation'
    for path,digest in read(HERE/f'{phase}_COMPLETE.receipt.json')['files'].items():assert sha(path)==digest,path
    base=read(HERE/'BASE_PROTOCOL.json');names=base['controller_order'];states=read(HERE/'NEW_EVAL_STATES.json')
    labels=np.full((62,24,14,2),-2,np.int8);uids=np.full(labels.shape,'',dtype='U96')
    events={k:np.zeros_like(labels) for k in ('collision','timeout','unknown')}
    traces=np.full(labels.shape,'',dtype='U512');trace_hashes=np.full(labels.shape,'',dtype='U64')
    with rollout_db.connect(True) as db:
        for c in read(HERE/f'{phase}_CASES.json'):
            i=names.index(c['base_controller_name']);s=c['phase_state_index'];a=c['candidate_index'];q=c['seed']
            assert labels[i,s,a,q]==-2
            row=db.execute('SELECT success,numerical_failure,collision,timeout FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            assert row is not None
            labels[i,s,a,q]=-1 if row['numerical_failure'] else row['success'];uids[i,s,a,q]=c['rollout_uid']
            raw=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
            if raw.exists():
                record=read(raw);assert record['case']==c
                traces[i,s,a,q]=record['trace_path'];trace_hashes[i,s,a,q]=record['trace_sha256']
            for e in events:events[e][i,s,a,q]=row['numerical_failure'] if e=='unknown' else row[e]
    assert not np.any(labels==-2)
    target=HERE/'NEW_EVAL_BUNDLE.npz';assert not target.exists()
    np.savez_compressed(target,names=np.asarray(names),types=np.asarray([base['types'][n] for n in names]),
        state_uids=np.asarray([s['state_uid'] for s in states]),labels=labels,uids=uids,traces=traces,trace_hashes=trace_hashes,**events)
    seal('NEW_EVAL_BUNDLE.receipt.json',[target,HERE/'assemble_new_evaluation.py',HERE/f'{phase}_COMPLETE.receipt.json',
        HERE/'NEW_EVAL_CHOICES.receipt.json'],all_fourteen_actions_two_seeds=True,unknown_not_reexecuted=True)
    print('Complete new-state outcome bundle sealed',flush=True)


if __name__=='__main__':main()
