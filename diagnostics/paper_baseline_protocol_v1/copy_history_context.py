"""Read immutable historical identity rows from the official database only."""
import guard
from guard import HERE,ROOT
import datetime
import hashlib
import json
import sqlite3


def main():
    index=json.loads((HERE/'OURS_HISTORY_INDEX.json').read_text())
    database=ROOT/'shared_rollout_db/rollout.sqlite'
    destination=HERE/'inputs/ours_history/db_context.jsonl'
    destination.parent.mkdir(parents=True,exist_ok=True)
    rows={};missing=[]
    connection=sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    try:
        connection.execute('BEGIN')
        for uid,item in sorted(index['records'].items()):
            result=item['result']
            stored=connection.execute('SELECT state_uid,controller_uid,success,collision,timeout,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(uid,)).fetchone()
            if stored is None:
                missing.append(uid)
                continue
            assert not stored['conflict_quarantined'],('Quarantined historical source',uid)
            assert all(int(stored[k])==int(result[k])for k in ('success','collision','timeout','numerical_failure'))
            assert stored['state_uid']==result['state_uid'] and stored['controller_uid']==result['controller_uid']
            state=connection.execute('SELECT scenario_uid,physical_state_json,goals_geometry_json,provenance_json FROM state WHERE state_uid=?',(result['state_uid'],)).fetchone()
            controller=connection.execute('SELECT scenario_uid,config_json FROM controller_config WHERE controller_uid=?',(result['controller_uid'],)).fetchone()
            assert state and controller and state['scenario_uid']==controller['scenario_uid']==result['scenario_uid']
            rows[uid]=dict(source_rollout_uid=uid,official_rollout=dict(stored),state=dict(state),controller=dict(controller))
    finally:
        connection.close()
    temporary=destination.with_suffix('.tmp')
    with temporary.open('w')as stream:
        for uid,row in rows.items():stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    if destination.exists():
        assert destination.read_bytes()==temporary.read_bytes(),'Historical identity metadata changed'
        temporary.unlink()
    else:temporary.replace(destination)
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),official_database=str(database),access='read_only_snapshot',
                copied_identity_rows=len(rows),missing_rollout_rows=missing,
                copy=str(destination.relative_to(HERE)),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                immutable_row_values_only=True,no_whole_active_database_hash=True,new_rollouts=0,
                compatibility_certified=False)
    (HERE/'OURS_HISTORY_CONTEXT_COPY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Official historical identity contexts copied',len(rows),'missing',len(missing),'; no execution compatibility claim.')


if __name__=='__main__':main()
