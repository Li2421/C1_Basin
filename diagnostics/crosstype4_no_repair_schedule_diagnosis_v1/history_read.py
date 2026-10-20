"""Read all prescribed history; inventory physical evidence before new work."""
from common import *
def main():
    required={
      'crosstype3_unseen_controller_schedule_v1':'REPORT.md FINAL_REVIEW.md HANDOFF.md GAPS.md FAILURE_ATTRIBUTION.md AXIS_TABLE.md ZOO_MANIFEST.csv UNSEEN_CONTROLLER_CONFIGS.json UNSEEN_CAPABILITY.json unseen_models.py controllers.py zoo_runtime.py new_execution.py progress.py design.py candidates.py CANDIDATES.json STATE_INPUTS.npz TRAINING_ACTIONS.npz critic_model.py train_critic.py select_models.py SELECTED_MODELS.json inference.py new_choices.py FROZEN_PIPELINE.json NEW_EVAL_STATES.json NEW_EVAL_CHOSEN_ACTIONS.json NEW_EVAL_CHOICES.npz KNOWN_EVAL_BUNDLE.npz UNSEEN_EVAL_BUNDLE.npz DIAGNOSTIC_BUNDLE.npz diagnostic_CASES.json statistics_pipeline.py TRAINING_BUNDLE.npz CONTROLLER_FEATURES.npz NEW_TYPE_TIMING.json',
      'crosstype1c_two_eta_schedule_v1':'REPORT.md HANDOFF.md NEW_EVAL_STATES.json NEW_EVAL_CHOSEN_ACTIONS.json NEW_EVAL_BUNDLE.npz ZOO_MANIFEST.csv TRAINING_REFERENCES.npz',
      'crosstype2_mixed_zoo_selector_v1':'mixed_execution.py zoo_design.py mixed_runtime.py',
      'axis4a_geometry_horizon_v1':'progress.py',
      'orthoflow3_generator_critic_v1':'normalization.json generator/seed41/checkpoint.msgpack'}
    paths=[D/k/name for k,files in required.items() for name in files.split()]+[ROOT/'docs/generalization_axes_protocol_20261015.md',ROOT/'AGENTS.md',ROOT/'shared_rollout_db/README.md',ROOT/'docs/prompt_round37_blackbox_addendum_20261016.md']
    for record in read(HERE/'BASE_PROTOCOL.json')['profiles'].values():
        checkpoint=record['payload'].get('flow_checkpoint_path')
        if checkpoint:paths.append(Path(checkpoint))
    rows=[]
    for p in sorted(set(paths)):
        assert not any(part.startswith(('decentralized_','teacher_student_','multiscene_')) for part in p.parts)
        row=dict(path=str(p),sha256=sha(p),size=p.stat().st_size)
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:row['arrays']={k:dict(shape=z[k].shape,dtype=str(z[k].dtype)) for k in z.files}
        elif p.suffix=='.json':x=read(p);row['entries']=len(x) if isinstance(x,(list,dict)) else None
        elif p.suffix in ('.md','.py','.csv'):row['lines']=len(p.read_text().splitlines())
        rows.append(row)
    write(HERE/'HISTORY_READ_INVENTORY.json',rows)
    write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),required=required,reuse=[dict(source='crosstype3',compatible='All122 controller profiles, generator proposals/h/c on48 training and48 prior evaluation states, exact26-action draws/labels for104, all24-dimensional features, frozen choices forA1/A2, fresh-state labels and diagnostic labels.',missing='A switch variants and56 ordered progress pairs absent except exact caches; new18 lack old48×26 training labels; new zero/progress actions require exact preflight.'),dict(source='crosstype1c',compatible='All14 Flow profiles match current profiles; frozenη pairs,24 prior evaluation states and paired reference labels reusable; original62 training safety seed0–1 exists.',missing='A fixed-time and progress variants not fully evaluated; use official UID matches, never encode-only cache invalidation.'),dict(source='round35/round37',compatible='Original progress.py unchanged; p*=.9681620884830543 from observed500 calibration choice, no recalibration.',missing='No new threshold selection or controller/generator training authorized.'),dict(source='runtime',compatible='37 copied files with hashes/diffs; no historical constructor run in old directory.',missing='New-round16 replay and execution/early-termination equivalence gates before reuse/new labels.')],all_new_results_development=True,new_physics_only_for_missing_after_all_reuse_checks=True))
    seal('HISTORY_AUDIT.receipt.json',[HERE/'HISTORY_AUDIT.json',HERE/'HISTORY_READ_INVENTORY.json',HERE/'history_read.py'],before_new_experiments=True)
    print('Historical sources inventoried:',len(rows),flush=True)
if __name__=='__main__':main()
