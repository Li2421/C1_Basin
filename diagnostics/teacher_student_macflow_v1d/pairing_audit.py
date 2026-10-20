from core import *
from audit_utils import source_inventory
from registration import connect,canonical

def main():
    states=read(HERE/'EVAL_STATES.json');old=read(D/'decentralized_pipeline_v1/main_protocol.json')['states'];assert len(states)==len(old)==32
    for s,t in zip(states,old):
        assert s['study_index']==t['study_index'] and s['uid']==t['uid']
        for k in ['positions','velocities','goals']:assert np.asarray(s['physical'][k],np.float64).tobytes()==np.asarray(t['physical'][k],np.float64).tobytes()
    assert sha(HERE/'EVAL_STATES.json')==sha(D/'teacher_student_macflow_v1c/EVAL_STATES.json');assert sha(HERE/'TUNING_STATES.json')==sha(D/'teacher_student_macflow_v1c/TUNING_STATES.json')
    lookup={(c['base_controller_name'],c['state_index'],c['seed'],c['arm']):c for c in read(D/'teacher_student_macflow_v1c/main_CASES.json')};cases=read(HERE/'C_CASES.json');reused=[]
    for c in cases:
        arm='F10' if c['arm']=='K10' else 'F13' if c['arm']=='K13' else c['arm']
        if c['kind']!='student' or arm not in ['F10','F13','S300','S500','safety baseline']:continue
        p=lookup[(c['base_controller_name'],c['state_index'],c['seed'],arm)]
        for k in ['rollout_uid','state_uid','eta_uid','controller_uid','seed_key']:assert c[k]==p[k]
        reused.append(c['rollout_uid'])
    assert len(reused)==4480
    with connect(True) as db:
        for c in cases:
            s=next(s for s in states if s['state_uid']==c['state_uid']);r=db.execute('SELECT physical_state_json FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone();assert json.loads(r[0])==s['physical']
    write(HERE/'PAIRING_AUDIT.json',dict(utc=utc(),passed=True,evaluation_states=32,round28_physical_float64_exact=True,RNG_state_uid_alias_preserved=True,tuning_states_round30c_byte_exact=True,student_legacy_menu_cases_exact_identity=len(reused),student_legacy_menu_unique=len(set(reused)),DB_physical_exact=True,new_rollouts=0));source_inventory('imports_pairing_audit.json')
if __name__=='__main__':main()
