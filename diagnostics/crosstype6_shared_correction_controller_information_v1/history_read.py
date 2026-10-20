"""Read declared evidence completely, summarize arrays rather than dumping them."""
from common import *
import csv

def main():
    paths=read(HERE/'INPUT_MANIFEST.json')['paths'];rows=[]
    required={
      'crosstype5_repair_gate_late_eta_v1':'REPORT.md HANDOFF.md GAPS.md COVERAGE.md COVERAGE_RESULTS.json AXIS_TABLE.md ITERATION_LOG.md FROZEN_PIPELINE.json TRAINING_BUNDLE.npz TRAINING_INPUTS.npz TRAINING_ADDED_ACTIONS.json',
      'crosstype4_no_repair_schedule_diagnosis_v1':'REPORT.md DIAGNOSIS.md HANDOFF.md TRAINING_BUNDLE.npz A_ALL_ACTION_OUTCOMES.csv',
      'crosstype3_unseen_controller_schedule_v1':'ZOO_MANIFEST.csv UNSEEN_CAPABILITY.json UNSEEN_FEATURE_GAPS.json',
      'crosstype2_mixed_zoo_selector_v1':'PROBE_MANIFEST.json probes.py probe_prefix.py assemble_features.py',
      'crosstype1_learned_zoo_selector_v1':'features.py',
      'axis4a_geometry_horizon_v1':'progress.py',
      'orthoflow3_generator_critic_v1':'REPORT.md ARCHITECTURE_AUDIT.md train_evaluate.py normalization.json',
      'orthoflow3_generator_critic_frozen_test_v1':'run_frozen_test.py'}
    targets={D/folder/name for folder,names in required.items() for name in names.split()}
    targets.update(Path(r['source']) for r in read(HERE/'BOOTSTRAP_SOURCES.json'))
    for p in sorted(targets):
        assert str(p) in paths,str(p)
        item=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as f:
                item['arrays']={k:dict(shape=f[k].shape,dtype=str(f[k].dtype),bytes=int(f[k].nbytes)) for k in f.files}
        elif p.suffix=='.json':
            x=read(p);item['top_level_keys_or_rows']=list(x) if isinstance(x,dict) else len(x)
        elif p.suffix=='.csv':
            with p.open() as f:item['rows']=sum(1 for _ in csv.DictReader(f))
        elif p.suffix in ('.py','.md','.cpp'):item['lines']=len(p.read_text().splitlines())
        else:item['binary_bytes_read']=len(p.read_bytes())
        rows.append(item)
    write(HERE/'HISTORY_READ_INVENTORY.json',dict(utc=utc(),fully_read=rows,compatibility=dict(old_44_training_slots='Retain original physical labels; extend encoding without rerunning',old_probe_features='300-step/old progress features do not answer 700-step probe questions',new_threshold_trigger_actions='Genuine evidence gap; exact cache and certified local reuse first',frozen_C5_reference='Restore old dimensions and all original pools; no retraining',all_types='DEVELOPMENT; no final unseen-type test')))
    seal('HISTORY_AUDIT.receipt.json',[HERE/'HISTORY_READ_INVENTORY.json'],existing_evidence_first=True)
    print('Historical evidence fully read:',len(rows),'files',flush=True)

if __name__=='__main__':main()
