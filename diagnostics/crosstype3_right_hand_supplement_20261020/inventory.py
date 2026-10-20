"""Read-only historical evidence inventory; excludes all forbidden directory trees."""
from common import *

def main():
    sources={
      'crosstype1c_two_eta_schedule_v1':'REPORT.md HANDOFF.md GAPS.md ITERATION_LOG.md AXIS_TABLE.md PREREGISTRATION.md candidates.py CANDIDATES.json STATE_INPUTS.npz TRAINING_STATES.json TRAINING_ACTIONS.npz TRAINING_BUNDLE.npz BASE_PROTOCOL.json CONTROLLER_FEATURES.npz critic_model.py train_critic.py select_models.py CRITIC_TRAINING_PLAN.json SELECTED_MODELS.json MAIN_METHOD_SELECTION.json inference.py new_choices.py FROZEN_PIPELINE.json statistics_pipeline.py confirmatory_statistics.py NEW_EVAL_STATES.json resource_guard.py',
      'crosstype2_mixed_zoo_selector_v1':'REPORT.md HANDOFF.md GAPS.md ZOO_MANIFEST.csv mixed_controllers.py mixed_runtime.py mixed_execution.py zoo_design.py conservative_reward.py CONTROLLER_FEATURES.npz assemble_features.py probes.py probe_prefix.py NEW_EVAL_BUNDLE.npz PROBE_TRAJECTORY_INDEX.json',
      'crosstype1_learned_zoo_selector_v1':'ZOO_MANIFEST.csv features.py train_bc.py zoo_models.py TRAINING_CONFIGS.json',
      'blackbox_families_v1':'spacetime_astar.py fast_astar.py astar_kernel.cpp corrected_astar_reference.py CORRECTED_ASTAR_ENGINEERING_TESTS.json DESIGN_CONFIG.json PILOT_STATES.json train_policies.py learned.py',
      'other_controller_classes_v1':'controllers.py',
      'orthoflow3_generator_critic_v1':'normalization.json generator/seed41/checkpoint.msgpack',
      'orthoflow3_generator_critic_frozen_test_v1':'run_frozen_test.py',
      'axis4a_geometry_horizon_v1':'progress.py'}
    paths=[D/sub/f for sub,files in sources.items() for f in files.split()]+[ROOT/'new_benchmark_common/dataset.py']
    paths += [Path(c['manifest']) for c in read(HERE/'TRANSFORMER_TRAINING_CONFIGS.json')]
    base=read(HERE/'BASE_PROTOCOL.json')
    for r in base['profiles'].values():
        p=r['payload'].get('flow_checkpoint_path')
        if p:
            paths.append(Path(p));normal=Path(p).parent/'NORMALIZATION.json'
            if normal.exists():paths.append(normal)
    rows=[]
    for p in sorted(set(paths)):
        assert any(p.resolve().is_relative_to(root) for root in (ROOT,FIELD,Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official')))
        assert not any(part.startswith(('decentralized_','teacher_student_','multiscene_')) for part in p.parts)
        entry=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:entry['arrays']={k:dict(shape=z[k].shape,dtype=str(z[k].dtype)) for k in z.files}
        elif p.suffix=='.json':
            value=read(p);entry['json_kind']=type(value).__name__;entry['entries']=len(value) if isinstance(value,(dict,list)) else None
        elif p.suffix in ('.py','.md','.csv','.cpp'):entry['text_lines']=len(p.read_text().splitlines())
        rows.append(entry)
    write(HERE/'HISTORY_READ_INVENTORY.json',rows)
    write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),sources=sources,reuse=[dict(source=str(D/'crosstype1c_two_eta_schedule_v1'),compatibility='All62 payloads and training state/action draws exact; both reproduction gates passed; DB UID assertion before any new labels.',missing='No new physics for exact old labels, unknown preserved'),dict(source=str(D/'crosstype2_mixed_zoo_selector_v1'),compatibility='All104 eligible profiles reused unchanged; corrected A* / MPPI / distributed CEM reproduced bitwise; first24 mechanism features only.',missing='42 nonlegacy controllers need exact two-eta sparse label preflight; old five-generator diagonal labels reused where exact. RHR excluded.'),dict(source=str(D/'ring_exchange_stage1'),compatibility='Corresponding frozen Flow normalizations and frames, existing train/dev splits.',missing='No historical Transformer policy of the specified architecture exists; six fixed offline trainings required. Test trajectories never loaded.'),dict(source='Three new types',compatibility='New nominal code only; reproduced physical/safety/runtime loops unchanged.',missing='Capability, termination-disabled mechanisms and one frozen new-state evaluation necessary; gate correction success rates blinded.')],source_and_input_inventory='HISTORY_READ_INVENTORY.json',feature_reuse_condition='Recompute4 original and4 extra controllers bitwise from saved no-outcome traces before reuse',protected_trees_pruned=True))
    seal('HISTORY_AUDIT.receipt.json',[HERE/'HISTORY_AUDIT.json',HERE/'HISTORY_READ_INVENTORY.json',HERE/'inventory.py'],before_new_label_submission=True)

if __name__=='__main__':main()
