"""Read-only input identity audit; writes only the new diagnostic directory."""
from pathlib import Path
import hashlib,json,datetime
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
F=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
H=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
old=ROOT/'diagnostics/tt_ff_correction_realization_20261017_v1'
prior=json.loads((old/'validation_manifest.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={**prior['source_hashes'],**prior['loaded_module_hashes']}
files.update({str(p):sha(p) for p in (old/'validation_manifest.json',old/'geometry_data.json',
    old/'tt_ff_action_validation.csv',old/'ff_substep_validation.csv',old/'trajectory_metrics.json',
    H/'design.py',F/'source/scripts/giveway_initial_state.py',F/'family_study_v2/protocol.json')})
for p,h in files.items():assert sha(Path(p))==h,p
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(F),
    historical_root=str(H),source_and_prior_evidence_hashes=files,
    checkpoint=prior['checkpoint'],checkpoint_sha256=prior['checkpoint_sha256'],
    runtime_config=prior['runtime_config'],cbf=prior['cbf'],noise_future_root=prior['future_root'],
    evidence_reuse=[dict(path=str(old),compatibility='matching frozen source/config/checkpoint; immutable action/substep replay evidence',
        gap='no whole-region reachability/progress certificate; witness full episodes UNKNOWN'),
        dict(path=str(H/'confirmation_manifest.json'),compatibility='same frozen hard-state panel and K16; source hashes verified',
        gap='sampled outcomes only, not region theorem')],
    execution_policy=dict(new_rollouts=0,new_training=0,source_mutations=0,
        allowed='geometry arithmetic, saved-trajectory analysis; bounded no-step probes only after cost announcement'),
    agent_dependency='A completed geometry contract -> independent B/C -> D integrates completed A/B/C',
    manuscript='No original Proposition 5 / equation (15) / Part 3/4 text located yet; user asked asynchronously')
(HERE/'input_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('Verified',len(files),'frozen source/evidence files')
