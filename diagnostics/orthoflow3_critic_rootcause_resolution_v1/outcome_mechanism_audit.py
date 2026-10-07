"""Read completed source rollout telemetry only; no new task or model input."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .controller_diversity import OUT as SOURCE,read,write
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite
from shared_rollout_db.src.rollout_db import connect,uid
OUT=SOURCE.parent

def main():
    entries=read(SOURCE/'dataset_DB_keys.json');allowed={rid for r in entries for rid in r['rollout_uids']}
    mapping={};files=set();ids=sorted(allowed)
    with connect(True) as db:
        for start in range(0,len(ids),500):
            batch=ids[start:start+500]
            for r in db.execute('SELECT rollout_uid,original_source_file,raw_record_hash FROM rollout WHERE rollout_uid IN ('+','.join('?' for _ in batch)+')',batch):
                mapping[r['raw_record_hash']]=r['rollout_uid'];files.add(r['original_source_file'])
    found={}
    from diagnostics.orthoflow3_controller_training_repair_v1.data import digest
    for path in sorted(files):
        if not path or not Path(path).is_file():continue
        with open(path) as f:
            for line in f:
                obj=json.loads(line);r=obj.get('record',obj);h=digest(r)
                if h not in mapping:continue
                rid=mapping[h]
                if rid in found:continue
                found[rid]=r
    groups=defaultdict(list)
    for r in found.values():
        if r.get('numerical_failure'):continue
        groups[(r['controller_uid'],bool(r['success']))].append(r)
    rows=[]
    for (controller,success),records in sorted(groups.items()):
        def arr(key):return np.array([r[key] for r in records if r.get(key) is not None],float)
        goal=arr('terminal_goal_error');clear=arr('minimum_agent_clearance');project=arr('second_projection_active_fraction')
        modes={k:sum(r.get('mode_signature')==k for r in records) for k in ('mixed','cw','ccw','stationary')}
        rows.append(dict(controller_uid=controller,success=success,records=len(records),**modes,
            terminal_goal_error_median=float(np.median(goal)) if len(goal) else None,
            failure_terminal_error_above_1_fraction=float(np.mean(goal>1)) if len(goal) else None,
            agent_clearance_below_001_fraction=float(np.mean(clear<.01)) if len(clear) else None,
            second_projection_active_mean=float(project.mean()) if len(project) else None))
    csvwrite(OUT/'source_outcome_mechanisms.csv',rows)
    valid=[r for r in found.values() if not r.get('numerical_failure')]
    result=[]
    for mode in ('mixed','cw','ccw','stationary'):
        rr=[r for r in valid if r.get('mode_signature')==mode]
        result.append(dict(realized_trajectory_signature=mode,records=len(rr),success_rate=float(np.mean([r['success'] for r in rr])) if rr else None))
    write(OUT/'source_outcome_mechanism_audit.json',dict(expected=len(allowed),found=len(found),raw_files=len(files),by_signature=result,
        interpretation='Post-outcome trajectory signature is diagnostic ONLY; not a mode input, classifier target, or candidate construction rule. Summaries distinguish late tracking versus sustained coordination failure, not establish a predictive feature.',
        target_confirmation_labels_used=False,new_rollouts=0))
    print(result)

if __name__=='__main__':main()
