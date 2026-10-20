from core import *
from audit_utils import source_inventory
import sqlite3
A=D/'teacher_student_macflow_v1';B=D/'teacher_student_macflow_v1b'
def main():
    assert (HERE/'INTEGRITY_START.json').exists()
    seal('PREREGISTRATION.receipt.json',[HERE/'PREREGISTRATION.md',ROOT/'docs/prompt_round30c_regate_and_tuned_rhr_20261015.md'],before_new_metrics=True,before_new_rollouts=True)
    students={};audit=[];new=[0,2,3,4,5,6,9,11]
    for i,r in enumerate(read(HERE/'TEACHERS.json')):
        origin=A if i in [8,10,12,13] else B;ck=origin/'students'/r['name']/'best.pkl';complete=read(ck.parent/'COMPLETE.json');assert sha(ck)==complete['checkpoint_sha256'];assert sha(r['checkpoint'])==r['checkpoint_sha256'];data=A/'data'/r['name'];ready=read(data/'READY.json')
        for f in ['dev_local.npy','dev_teacher.npy','normalization.npz']:assert sha(data/f)==ready['files'][f]
        students[str(i)]=dict(index=i,name=r['name'],checkpoint=str(ck),checkpoint_sha256=sha(ck),origin=str(origin),best_step=complete['best_step'],previously_valid=i not in new)
    from shared_rollout_db.src.rollout_db import connect
    with connect(True) as db:
        for i in new:
            ck=students[str(i)]['checkpoint_sha256'];rows=db.execute('SELECT r.rollout_uid FROM rollout r JOIN controller_config c USING(controller_uid) WHERE c.flow_checkpoint_sha256=?',(ck,)).fetchall();assert not rows,('Previously closed-loop labels for new candidate',i)
            audit.append(dict(index=i,student=students[str(i)],existing_rollouts=0))
    write(HERE/'STUDENTS.json',students);write(HERE/'REGATE_INDICES.json',new);write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),candidates=audit,reuse='all original6 seven-arm labels and frozen choices; teacher supplementary labels; all14 frozen checkpoints and dev data; round30b equivariance diagnostics',gap='filtered-cosine not previously saved; eight candidates have no closed-loop evidence; new independent RHR tuning states/grid and tuned-main labels absent',training_required=False))
    files=['core.py','pipeline.py','episode_frozen.py','episode_extended.py','actions.py'];assert all(sha(HERE/f)==sha(B/f) for f in files)
    gate=read(B/'ROUND28_REPLAY_GATE.json');assert gate['passed'] and gate['completed']==16;write(HERE/'ROUND28_REPLAY_GATE.json',dict(passed=True,completed=16,reused_from=str(B/'ROUND28_REPLAY_GATE.json'),sha256=sha(B/'ROUND28_REPLAY_GATE.json'),unchanged_components={f:sha(HERE/f) for f in files}));seal('ROUND28_REPLAY_GATE.receipt.json',[HERE/'ROUND28_REPLAY_GATE.json']+[HERE/f for f in files],reuse=True)
    source_inventory('imports_prepare.json')
if __name__=='__main__':main()
