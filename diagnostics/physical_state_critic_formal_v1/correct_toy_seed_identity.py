"""One-time outcome-blind correction: include Toy rollout_id in continuation key."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from shared_rollout_db.src.rollout_db import canonical, connect, uid
from diagnostics.physical_state_critic_formal_v1 import loso_worker

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    target=HERE/'loso'/'toy_seed_identity_correction_audit.json'
    if target.exists():
        print(target.read_text());return
    changes=[]
    for split in ('train','validation','test'):
        panel=HERE/'loso'/'toy_give_way'/split
        proposals=read(panel/'proposals.json')['states']
        path=panel/'cases.json'
        cases=read(path)
        old_hash=sha(path)
        assert all('rollout_id' not in json.loads(c['seed_key']) for c in cases)
        for name in ('cases.json','planned_rollouts.json','missing_cases.json',
                     'cache_preflight.json','preflight_exact_audit.json'):
            source=panel/name
            if source.exists():shutil.copy2(source,panel/(name+'.superseded_no_rollout_id'))
        for c in cases:
            rollout_id=proposals[c['state_index']]['rollout_id']
            c['seed_key']=canonical(dict(future_index=c['future_index'],rollout_id=rollout_id))
            identity=dict(state=c['state_uid'],eta=c['eta_uid'],
                          controller=c['controller_uid'],seed=c['seed_key'])
            c['rollout_uid']=uid('roll',identity)
        assert len({c['rollout_uid'] for c in cases})==len(cases)
        path.write_text(json.dumps(cases,indent=2,sort_keys=True)+'\n')
        trials=4 if split=='train' else 16
        requests=[]
        for i in range(0,len(cases),trials):
            block=cases[i:i+trials]
            assert len({(c['state_uid'],c['eta_uid'],c['controller_uid']) for c in block})==1
            requests.append(dict(state_uid=block[0]['state_uid'],eta_uid=block[0]['eta_uid'],
                                 controller_uid=block[0]['controller_uid'],
                                 seed_keys=[c['seed_key'] for c in block]))
        (panel/'planned_rollouts.json').write_text(json.dumps(dict(requests=requests),indent=2,sort_keys=True)+'\n')
        subprocess.run([sys.executable,'-m','shared_rollout_db.plan',
                        '--manifest',str(panel/'planned_rollouts.json'),
                        '--output',str(panel/'cache_preflight.json')],check=True,
                       stdout=(panel/'preflight_stdout.json').open('w'))
        with connect(True) as db:
            old_rows=sum(db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is not None
                         for c in cases)
        assert old_rows==0,'Corrected identities must be fresh'
        (panel/'missing_cases.json').write_text(json.dumps(cases,indent=2,sort_keys=True)+'\n')
        audit=dict(scene='toy_give_way',split=split,slots=len(cases),exact_valid=0,
                   exact_numerical_unknown=0,missing=len(cases))
        (panel/'preflight_exact_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        changes.append(dict(split=split,old_cases_sha256=old_hash,
                            corrected_cases_sha256=sha(path),cases=len(cases),
                            previous_exact_reuse=64 if split=='validation' else 0,
                            corrected_exact_reuse=0,eta_candidates_unchanged=True,
                            physical_states_unchanged=True))
    previous=HERE/'loso_source_work_items.json'
    old_hash=sha(previous)
    new=loso_worker.items('source')
    assert len(new)==1296
    previous.write_text(json.dumps(new,separators=(',',':'))+'\n')
    audit=dict(schema='toy_loso_seed_identity_correction_v1',
               issue='Toy frozen Flow folds rollout_id into its t0 and future PRNG keys; cache seed identity previously omitted rollout_id',
               corrected_before_any_loso_test_labels_or_checkpoint_training=True,
               affected_jobs_cancelled=[6509],
               old_pilot_and_partial_raw_evidence_retained=True,
               old_journals_and_merged_rows_retained_but_excluded_by_corrected_case_uids=True,
               source_work_items_old_sha256=old_hash,
               source_work_items_corrected_sha256=sha(previous),
               changes=changes)
    target.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
