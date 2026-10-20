"""Explicitly scoped source/data inventory; forbidden prefixes never opened."""
from core import *


def main():
    groups={
      'crosstype1b_generator_critic_pipeline_v1':('REPORT.md','HANDOFF.md','GAPS.md','ITERATION_LOG.md','AXIS_TABLE.md','AXIS_METRICS.json','CANDIDATES.json','condition_candidates_v2.py','critic_model.py','action_only_model.py','train_critic.py','train_action_only.py','select_models.py','select_models_v2.py','select_matched_v2b.py','new_evaluation_choices.py','statistics_pipeline.py','frozen_networks.py','resource_guard.py','FROZEN_PIPELINE.json','TRAINING_BUNDLE.npz','BASE_PROTOCOL.json','CONTROLLER_FEATURES.npz'),
      'orthoflow3_generator_critic_v1':('REPORT.md','ARCHITECTURE_AUDIT.md','train_evaluate.py','normalization.json','generator/seed41/checkpoint.msgpack','critic/seed23/checkpoint.msgpack'),
      'orthoflow3_generator_critic_frozen_test_v1':('run_frozen_test.py',),
      'crosstype1_learned_zoo_selector_v1':('REPORT.md','HANDOFF.md','ZOO_MANIFEST.csv','features.py','assemble.py','initial_actions.py','DEVELOPMENT_FEATURES.npz','LOCKED_FEATURES.npz','IMPLICIT_IMPLEMENTATION_GAP.json','IMPLICIT_SYNTHETIC_TEST.json','IMPLICIT_FULLCOV_SYNTHETIC_TEST.json','IMPLICIT_ACTIVATION_TEST.json','IMPLICIT_LANDSCAPE_DIAGNOSIS.json','training_env.py','ippo.py','train_ippo.py','IPPO_TRAINING_CONFIGS.json','TRAINING_ENV_GATE.json'),
      'blackbox_families_v1':('REPORT.md','GAPS.md','DESIGN_CONFIG.json','controllers.py','episode.py','EVALUATION_LABELS.npz','PILOT_STATES.json','TUNING_STATES.json','EVAL_STATES.json','FIXED_RIGHT_HAND_RECOMPUTATION.json','corrected_astar_reference.py','CORRECTED_ASTAR_ENGINEERING_TESTS.json','ASTAR_OBLIQUE_UNIT_TEST.json','IMPLEMENTATION_GAP_ANALYSIS_AMENDMENT.json','fast_astar.py','astar_kernel.cpp'),
      'blackbox_addendum_v1':('REPORT.md','continuing_rhr.py','RIGHT_HAND_GRID_RESULTS.json','RIGHT_HAND_SELECTED.json','controllers.py'),
      'axis2_controller_v1':('controllers.py',),
      'other_controller_classes_v1':('controllers.py',),
      'axis4a_geometry_horizon_v1':('progress.py',),
    }
    entries=[];missing=[]
    paths=[ROOT/'AGENTS.md',ROOT/'shared_rollout_db/README.md',ROOT/'docs/generalization_axes_protocol_20261015.md',ROOT/'docs/controller_generalization_audit_20261011.md',ROOT/'docs/prompt_round41_multiscene_decentralized_reinforcement_learning_20261017.md']
    for directory,names in groups.items():
        assert not directory.startswith(('decentralized_','teacher_student_','multiscene_'))
        paths.extend(D/directory/name for name in names)
    paths.extend(sorted((D/'blackbox_families_v1').glob('pilot_*_FAIR_RULES.json')))
    for p in paths:
        if not p.exists():missing.append(str(p));continue
        value=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:value['arrays']={k:dict(shape=z[k].shape,dtype=str(z[k].dtype)) for k in z.files}
            value['inspection']='Schema inventory; historical A* engineering outcomes not analyzed'
        elif p.suffix=='.json':
            data=read(p);value['structure']=dict(type=type(data).__name__,size=len(data) if isinstance(data,(dict,list)) else None,keys=list(data)[:20] if isinstance(data,dict) else None)
        elif p.suffix in ('.py','.md','.csv','.cpp'):
            text=p.read_text();value['lines']=len(text.splitlines())
        entries.append(value)
    write(HERE/'HISTORY_READ_INVENTORY.json',dict(utc=utc(),files=entries,missing=missing,forbidden_content_read=False,old_astar_engineering_evaluation_used_for_tuning=False))
    old=D/'crosstype1b_generator_critic_pipeline_v1'
    audit=dict(utc=utc(),reusable=[
      dict(question='Learned controller deployment and 13-candidate training labels',source=str(old),compatibility='62 original frozen checkpoints, exact 48 training states, unchanged physics/RNG/progress/wrapper; candidate 0–12 labels reused, including existing anchor seed1. Changing controller encoding does not invalidate labels.',gap='Only three RHR candidates; new RHR probe arm'),
      dict(question='Existing three MPC nominal sources',source=[str(D/n) for n in ('axis2_controller_v1','blackbox_addendum_v1','blackbox_families_v1')],compatibility='Round38 tuning16/evaluation32 have exactly the same physical layouts as the 48 training states; compare exact identities and RNG before reuse. Reproduction gates passed.',gap='Generator candidates and shortlisted RHR arms; fixed RHR/anchors reused where exact'),
      dict(question='Corrected A*',source=str(D/'blackbox_families_v1/corrected_astar_reference.py'),compatibility='Engineering source and unit evidence reusable; old 244 engineering evaluation labels explicitly excluded.',gap='Fresh identity, full unit tests, capability gate, probes, all candidate labels'),
      dict(question='Conservative PPO',source=str(D/'crosstype1_learned_zoo_selector_v1'),compatibility='Original training environment copied byte-identically; exact recipes, training/checkpoint states, seeds, optimizer, network and 20M budget retained. Only specified reward differs.',gap='Reward differs, so old checkpoint cannot answer new policy question; six new training runs necessary'),
      dict(question='Optional implicit policy',source=str(D/'crosstype1_learned_zoo_selector_v1/IMPLICIT_IMPLEMENTATION_GAP.json'),compatibility='Original two-mode criterion unchanged; five tested implementation/activation variants fail. No Ring-trained checkpoint exists.',gap='Optional type absent unless a documented implementation correction passes the unchanged gate; no replacement'),
      dict(question='Frozen generator and old pipeline',source=str(old/'FROZEN_PIPELINE.json'),compatibility='72-state proposal/original critic recomputation bitwise passed; old nine-type all-development model remains immutable baseline, only old C24 and first13 candidates.',gap='No old evidence evaluates locked new optimizer types on fresh states'),
    ],new_evaluation_states='Fresh seed interval and all-DB recoverable physical dedup; no prefreeze closed-loop outcomes',training_oracle='Training generator seed0 only; report required in-sample headroom, do not claim cross-fit there',missing_files=missing)
    write(HERE/'HISTORY_AUDIT.json',audit)
    seal('HISTORY_AUDIT.receipt.json',[HERE/'HISTORY_AUDIT.json',HERE/'HISTORY_READ_INVENTORY.json',HERE/'history_inventory.py'])
    print('Explicit history inventory:',len(entries),'files;',len(missing),'missing')


if __name__=='__main__':main()
