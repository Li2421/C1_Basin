from core import *
from registration import connect,eta_identity
from audit_utils import source_inventory
BASE=D/'decentralized_pipeline_v1';C=D/'teacher_student_macflow_v1c'
def main():
    paths=[BASE/'main_protocol.json',D/'controller_q_transfer_round3_v1/protocol.json',D/'controller_q_diversity_v1/protocol.json'];lists=[read(p)['eta'] for p in paths];a=np.asarray(lists[0],np.float64);assert a.shape==(16,3);assert all(a.tobytes()==np.asarray(b,np.float64).tobytes() for b in lists)
    with connect(True) as db:
        checks=[]
        for j,v in enumerate(a):
            eid,values,h=eta_identity(v);r=db.execute('SELECT * FROM eta WHERE eta_uid=?',(eid,)).fetchone();assert r is not None and r['canonical_hex']==h;checks.append(dict(index=j,eta=v,eta_uid=eid,canonical_hex=h,DB_exact=True))
        cases=read(C/'main_CASES.json');casechecks=[]
        for j in [10,13]:
            rs=[c for c in cases if c['arm']==f'F{j}'];assert len(rs)==14*64 and all(c['eta_uid']==checks[j]['eta_uid'] and np.asarray(c['eta_a'],np.float64).tobytes()==a[j].tobytes() for c in rs);casechecks.append(dict(index=j,existing_cases=len(rs),exact=True))
        rows=[dict(r) for r in db.execute('SELECT controller_uid,config_json FROM controller_config')];matches=[r for r in rows if any(t in r['config_json'].lower() for t in ['zero_nominal','zero_action','nonlearning_zero','straight_to_goal','nonlearning_goal'])]
    write(HERE/'K16_AUDIT.json',dict(utc=utc(),status='PASS',indexing='zero-based historical eta indices 10 and 13',sources={str(p):sha(p) for p in paths},eta=lists[0],identities=checks,F10_F13_checks=casechecks))
    write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),A=dict(searched_controller_rows=len(rows),lexical_matches=matches,code_search_roots=[str(BASE),str(C),str(D/'controller_q_transfer_round3_v1')],gap='No exact nonlearning Z/G profiles for unchanged D-local or centralized TT were found. New nominal semantics require A rollouts.',reuse='14 student S500: round30c main_CASES/DB_ROWS; 14 centralized teacher S500: round28 main_CASES and authoritative DB'),B=dict(reused_states=str(C/'TUNING_STATES.json'),old_grid=str(C/'RHR_GRID_RESULTS.json'),gap='Old trigger .01m/s cannot answer .05 speed, progress, or always-on grid. No fair-RHR exact profiles; new trials needed.'),C=dict(reuse='all student safety/F10/F13/S300/S500 from round30c and Z from A; other 14 K16 values are new exact execution semantics',K16=str(HERE/'K16_AUDIT.json')),sources={str(p):sha(p) for p in [C/'main_CASES.json',C/'main_DB_ROWS.json',C/'TUNING_STATES.json',BASE/'main_CASES.json']}))
    source_inventory('imports_history.json')
if __name__=='__main__':main()
