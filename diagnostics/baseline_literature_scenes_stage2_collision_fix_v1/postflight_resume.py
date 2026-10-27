from experiment import *
from shared_rollout_db.src.rollout_db import connect,canonical as db_canonical
from audit import V2

def main():
    assert (HERE/'PART1_COMPUTATION_READY.json').exists()
    rows=[]
    for folder in HERE.glob('lit_*/raw'):rows.extend(folder.glob('*.json'))
    errors=[];unknown=0;aliases=0
    with connect(True) as db:
        for p in rows:
            r=read(p);stored=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if stored is None:errors.append(dict(path=str(p),reason='missing_database_row'));continue
            keys=('state_uid','eta_uid','controller_uid','seed_key','success','collision','timeout','numerical_failure','outcome')
            if any(stored[k]!=r[k] for k in keys) or stored['conflict_quarantined']:errors.append(dict(path=str(p),reason='identity_or_outcome_mismatch'))
            if stored['raw_record_hash']!=hashlib.sha256(db_canonical(r).encode()).hexdigest():errors.append(dict(path=str(p),reason='raw_content_hash_mismatch'))
            unknown+=r['numerical_failure'];aliases+=bool(r.get('deterministic_source'))
        doorway=read(V2/'optional_dynamics_lit_doorway_CASES.json');door_missing=[]
        for c in doorway:
            found=db.execute('SELECT conflict_quarantined FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if not found or found['conflict_quarantined']:door_missing.append(c['rollout_uid'])
    pending=list((ROOT/'shared_rollout_db/journals/exp_baseline_literature_scenes_stage2_collision_fix_v1').glob('*.jsonl'))
    write(HERE/'DB_POSTFLIGHT_RESUME.json',dict(utc=utc(),passed=not errors and not door_missing and not pending,
        checked_new_rows=len(rows),unknown_rows_preserved=unknown,deterministic_seed_aliases=aliases,
        doorway_exact_reuse_requests=len(doorway),doorway_missing=door_missing,pending_journals=[str(p) for p in pending],errors=errors,
        skipped_initial_circle=7,skipped_cases_are_not_unknown=True))
    assert not errors and not door_missing and not pending,(errors[:3],door_missing[:3],len(pending))
    print('DB postflight passed:',len(rows),'new rows;',len(doorway),'doorway exact reuse',flush=True)
if __name__=='__main__':main()
