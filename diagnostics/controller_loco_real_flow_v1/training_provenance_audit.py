"""Verify frozen source-only normalization, controller exclusion and checkpoints."""
import json
import hashlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
BASE=Path(__file__).resolve().parent
import numpy as np
from diagnostics.controller_loco_real_flow_v1.train import build

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    rows=[]
    for stage,folder in [('stage_a',BASE/'fits'),('stage_b',BASE/'stage_b/fits')]:
        for file in sorted(folder.glob('fold*_*/complete.json')):
            m=json.loads(file.read_text());names,tr,va,held,wrong,cal=build(m['fold'],m['arm'],stage=='stage_b')
            norm=np.load(file.parent/'normalization.npz')
            mu=tr.mean((0,1));sd=tr.std((0,1));scale=np.where(sd>1e-6,sd,1.)
            assert names==m['source_controller_names'] and m['held_controller'] not in names
            assert np.array_equal(mu,norm['mean']) and np.array_equal(scale,norm['scale'])
            assert m['source_only_normalization'] and not m['test_labels_read'] and not m['jax_enable_x64']
            assert len(m['checkpoints'])==3 and m['fit_seeds']==[17,23,41]
            hashes={p:sha(p) for p in m['checkpoints']}
            rows.append(dict(stage=stage,fold=m['fold'],arm=m['arm'],source_count=len(names),
                             source_only_normalization_exact=True,held_controller_excluded=True,
                             checkpoints_sha256=hashes,selected_step=m['selected_step'],elapsed_seconds=m['elapsed_seconds']))
    assert len(rows)==45
    sourcepaths=[ROOT/'diagnostics/controller_q_transfer_round3_v1'/p for p in ['flow_adapter.py','state_generator.py','p1_features.py','p3_features.py']]
    sourcepaths += [ROOT/'diagnostics/controller_q_diversity_v1/runtime.py',ROOT/'diagnostics/orthoflow3_tt_ff_matched_learnability_v1/execution.py',ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/physical_inputs.npz',ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_LABELS.npz',ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_EXECUTION_PROTOCOL.json']
    files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sourcepaths}
    out=dict(status='PASS',fit_tasks=45,fit_seed_checkpoints=135,rows=rows,source_assets=files,
             total_fitting_task_seconds=sum(r['elapsed_seconds'] for r in rows),
             interpretation='No target labels are read here; exact source-only normalization and controller exclusion are verified from frozen input features and manifests.')
    (BASE/'TRAINING_PROVENANCE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'PASS','fit_tasks':45,'checkpoints':135}))

if __name__=='__main__':main()
