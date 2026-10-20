"""Full evidence check, single-process merge, postflight and immutable labels."""
from core import *
import subprocess

PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
ARMS=['F10','F13','F2','S200','S300','S400','S500','S600']

def call(args,log):
    with (HERE/log).open('w') as f:
        subprocess.run([PY,'-B',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def main(phase):
    assert phase in ('train','test')
    if phase=='train':train_labels_allowed()
    else:test_labels_allowed()
    p=read(HERE/'protocol.json');base=read(HERE/'FINAL_BASE_PROFILES.json')
    names=base['train_controller_order'] if phase=='train' else base['test_controller_order']
    states=p['state_groups'][phase];rows=read(HERE/f'{phase}_CASES.json')
    assert len(names)==18 and len(states)==(32 if phase=='train' else 48)
    assert len(rows)==18*len(states)*8*4
    label=np.full((18,len(states),8,4),-2,np.int8)
    index={n:i for i,n in enumerate(names)};unknown=[];outcomes={}
    for c in rows:
        path=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
        marker=HERE/phase/'journaled'/f"{c['rollout_uid']}.json"
        assert path.is_file() and marker.is_file(),c['rollout_uid']
        r=read(path)
        assert r['case']==c and r['protocol_sha256']==sha(HERE/'protocol.json')
        assert r['execution_code_sha256']==sha(HERE/'run.py') and r['jax_enable_x64'] is False
        assert read(marker)['raw_sha256']==sha(path)
        i=index[c['base_controller_name']];s=c['state_index'];a=ARMS.index(c['arm']);q=c['seed']
        assert label[i,s,a,q]==-2
        label[i,s,a,q]=-1 if r['numerical_failure'] else int(r['success'])
        if r['numerical_failure']:unknown.append(c['rollout_uid'])
        outcomes[r['outcome']]=outcomes.get(r['outcome'],0)+1
    assert not np.any(label==-2) and len({c['rollout_uid'] for c in rows})==len(rows)
    np.savez_compressed(HERE/f'{phase}_LABELS.npz',Y=label,controller_names=names,
                        state_uids=[s['state_uid'] for s in states],arms=ARMS)
    write(HERE/f'{phase}_COMPLETENESS_AUDIT.json',dict(status='PASS',utc=utc(),cases=len(rows),
          unknown=len(unknown),outcomes=outcomes,all_raw_journal_pairs_verified=True,
          labels_sha256=sha(HERE/f'{phase}_LABELS.npz')))
    call([str(HERE/'merge.py')],f'{phase}_merge.log')
    call(['-m','shared_rollout_db.plan','--manifest',str(HERE/f'{phase}_PLANNED_ROLLOUTS.json'),
          '--output',str(HERE/f'{phase}_POSTFLIGHT.json')],f'{phase}_postflight.log')
    call([str(HERE/'postflight.py'),phase],f'{phase}_unknown_audit.log')
    audit=read(HERE/f'{phase}_POSTFLIGHT_AUDIT.json')
    assert audit['status']=='PASS' and audit['unknown_present_in_db']==len(unknown)
    write(HERE/f'{phase}_DELIVERY_AUDIT.json',dict(status='PASS',utc=utc(),cases=len(rows),
          labels_sha256=sha(HERE/f'{phase}_LABELS.npz'),
          postflight_audit_sha256=sha(HERE/f'{phase}_POSTFLIGHT_AUDIT.json'),
          completeness_audit_sha256=sha(HERE/f'{phase}_COMPLETENESS_AUDIT.json'),
          unknown=len(unknown),true_absent=0))
    print(json.dumps(dict(phase=phase,status='PASS',cases=len(rows),unknown=len(unknown))))

if __name__=='__main__':main(sys.argv[1])
