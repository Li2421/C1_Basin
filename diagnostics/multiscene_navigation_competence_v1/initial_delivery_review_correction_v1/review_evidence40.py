"""Verify saved evidence and append-only journals; generate no trajectories."""
from common import *
from collections import Counter
from shared_rollout_db.src.rollout_db import connect

def main():
    assert (HERE/'FULL_NAVIGATION_PHASE_COMPLETE.json').exists()
    assert (HERE/'FULL_DESCRIPTIVE_EXECUTION_COMPLETE.json').exists()
    index=[];ids=[];unknown=[];source=ROOT/'shared_rollout_db/journals'/EXP
    for path in sorted(source.glob('*.jsonl.merged')):
        for line in path.read_text().splitlines():
            r=json.loads(line);rid=r.get('rollout_uid',r.get('case',{}).get('rollout_uid'));assert rid
            ids.append(rid)
            if r.get('trace_path'):
                p=Path(r['trace_path']);assert p.resolve().is_relative_to(HERE)
                assert sha(p)==r['trace_sha256'],str(p)
                index.append(dict(rollout_uid=rid,path=str(p),sha256=r['trace_sha256'],unlabeled=not any(k in r for k in ['success','outcome','termination'])))
            if r.get('numerical_failure'):unknown.append(rid)
    counter=Counter(ids);duplicates={k:v for k,v in counter.items() if v!=1};assert not duplicates,duplicates
    with connect(True) as db:
        quarantined=[r['rollout_uid'] for r in db.execute('SELECT rollout_uid FROM rollout WHERE experiment_uid=? AND conflict_quarantined=1',(EXP,))]
        authoritative=int(db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(EXP,)).fetchone()[0])
    assert not quarantined
    pointer=read(HERE/'HISTORICAL_SELECTOR_SOURCE40_CURRENT.json');assert sha(HERE/pointer['dataset'])==pointer['dataset_sha256']
    assert read(HERE/'FULL_G_STATUS.json')['status']=='GEOMETRY_GAP'
    forbidden_generated=[str(p) for pattern in ['full_expert_G_CASES.json','data39_G_round_*_CASES.json','navigation40_G_CASES.json','multiagent40_G_CASES.json','evaluation40_G_CASES.json'] for p in HERE.glob(pattern)];assert not forbidden_generated
    write(HERE/'SAVED_TRACE_DIGEST_INDEX40.json',index)
    write(HERE/'EVIDENCE_REVIEW40.json',dict(utc=utc(),passed=True,journal_records=len(ids),unique_rollout_ids=len(counter),duplicates=duplicates,trace_digests_verified=len(index),unknown_preserved=unknown,authoritative_rollout_rows=authoritative,unlabeled_summary_only=sum(r['unlabeled'] for r in index),quarantined=quarantined,G_new_training_or_corrective_cases=forbidden_generated,unknown_retries=0,review_generates_rollouts=0))
    seal('EVIDENCE_REVIEW40_COMPLETE.receipt.json',[HERE/'EVIDENCE_REVIEW40.json',HERE/'SAVED_TRACE_DIGEST_INDEX40.json',HERE/'review_evidence40.py'],no_new_rollouts=True,unknown_retries=0)

if __name__=='__main__':main()
