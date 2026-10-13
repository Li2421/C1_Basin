"""Register the frozen P2prime experiment omitted by the original design script."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from shared_rollout_db.src.rollout_db import canonical, connect, transaction

HERE=Path(__file__).resolve().parent
P=HERE/'p2prime'/'protocol.json'
EXP='exp_controller_q_transfer_round3_p2prime_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=json.loads(P.read_text());assert p['experiment_uid']==EXP
    assert sha(P)==sha(HERE/'P2PRIME_PROTOCOL.json')
    metadata=canonical(dict(max_new_rollouts=70000,max_cpu_workers=24,
                            actual_frozen_requests=67584,branch='P1 gate failed; held placement P2prime'))
    with connect() as db,transaction(db):
        old=db.execute('SELECT * FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone()
        if old:
            assert old['path']==str(HERE/'p2prime') and old['protocol_hash']==sha(P)
        else:
            assert db.execute('SELECT 1 FROM source_file WHERE experiment_uid=?',(EXP,)).fetchone() is None
            assert db.execute('SELECT 1 FROM rollout WHERE experiment_uid=?',(EXP,)).fetchone() is None
            db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                       (EXP,'Controller-Q Round3 P2prime held placement',str(HERE/'p2prime'),sha(P),sha(HERE/'p2prime_design.py'),metadata))
    print(json.dumps(dict(experiment_uid=EXP,path=str(HERE/'p2prime'),protocol_hash=sha(P),status='REGISTERED')))


if __name__=='__main__':main()
