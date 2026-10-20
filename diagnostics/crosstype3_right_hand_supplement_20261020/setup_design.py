"""Freeze data identities and controller grids before any new experiment."""
from common import *
import csv

def main():
    assert not (HERE/'OPENING_CONFIG.receipt.json').exists()
    c1=D/'crosstype1c_two_eta_schedule_v1';c2=D/'crosstype2_mixed_zoo_selector_v1'
    old=read(c1/'BASE_PROTOCOL.json');mixed=read(c2/'new_evaluation_protocol.json')
    with (c2/'ZOO_MANIFEST.csv').open() as f:rows=list(csv.DictReader(f))
    zoo=[r for r in rows if r['capability_passed']=='True'];names=[r['controller'] for r in zoo];types={r['controller']:r['type_code'] for r in zoo}
    assert len(names)==104 and len(set(types.values()))==16
    for n in old['controller_order']:assert mixed['profiles'][n]['payload']==old['profiles'][n]['payload'],n
    base={k:mixed[k] for k in ('profiles','eta','native_manifest')};base.update(controller_order=names,types=types)
    write(HERE/'BASE_PROTOCOL.json',base);write(HERE/'TRAINING_STATES.json',read(c1/'TRAINING_STATES.json'))
    learning=sorted(set(old['types'].values()));regression=[]
    for kind in learning:regression.extend([n for n in old['controller_order'] if old['types'][n]==kind][:2])
    extra=[n for n in names if n not in old['controller_order']];assert len(extra)==42 and len(regression)==18
    recipes=[(2,'base_u_dataset','base_u_macflow',0),(2,'base_u_v3_dataset','base_u_v3_macflow',1),(2,'base_u_plus_w_v3_dataset','base_u_plus_w_v3_macflow',2),(4,'base_u_dataset','base_u_macflow',3),(4,'base_u_v7_local_dataset','base_u_v7_local_macflow',4),(4,'base_u_v8_local_dataset','base_u_v8_local_macflow',5)]
    configs=[]
    for i,(layers,dataset,flow,seed) in enumerate(recipes):
        manifest=D/'ring_exchange_stage1'/dataset/'manifest.json';m=read(manifest);profile=old['profiles'][flow]['payload'];checkpoint=Path(profile['flow_checkpoint_path'])
        assert m['observation_shape']==[4,23] and m['action_shape']==[4,2]
        configs.append(dict(name=f'transformer_bc{i}',type='transformer_bc',family='learned',layers=layers,width=128,heads=4,feedforward=256,activation='gelu',pre_layernorm=True,dropout=0.,final_layernorm=False,seed=seed,validation_seed=seed+1000,manifest=str(manifest),manifest_sha256=sha(manifest),flow_checkpoint=str(checkpoint),flow_checkpoint_sha256=sha(checkpoint),frame=profile['observation_frame'],steps=5000,batch_size=256,optimizer='Adam',learning_rate=.0003,validation_interval=250,validation_examples=2048,checkpoint_selection='Lowest fixed dev joint Gaussian NLL; earliest exact tie, including initialization',log_std_bounds=[-5,1.5],test_split_never_opened=True,normalization='Exact corresponding Flow checkpoint stats, no refitting',source_recipe=flow))
    grids=dict(gradient_mpc=[[10,20,.05],[10,50,.05],[10,50,.2],[30,20,.05],[30,50,.05],[30,50,.2]],conflict_based_search=[[.05,10,64],[.05,10,512],[.05,25,512],[.1,10,64],[.1,10,512],[.1,25,512]])
    write(HERE/'TRANSFORMER_TRAINING_CONFIGS.json',configs);write(HERE/'UNSEEN_CONTROLLER_CONFIGS.json',dict(transformer_bc=configs,**grids))
    config=dict(utc=utc(),training_controllers=names,training_types=sorted(set(types.values())),training_controller_count=104,training_states=48,training_seed=[0],training_actions=26,old62_actions_reused_exactly=True,new42_actions_same_original_name_seed_rule=True,new48_seed_start=202610193001,new48_seed_stop_inclusive=202610193048,new_namespace='crosstype3_new_evaluation',new_eval_seeds=[0,1],new_eval_states_unseen=48,new_eval_states_training=24,regression_learning_order=regression,other42_order=extra,unseen_order=[f'{t}{i}' for t in ('transformer_bc','gradient_mpc','conflict_based_search') for i in range(6)],capability_states=str(D/'blackbox_families_v1/PILOT_STATES.json'),capability_arms=['safety_baseline','Sprog','F13'],capability_min_success=.15,no_repair_threshold=.98,min_instances_per_unseen_type=4,mechanism_dimension=24,mechanism_source_recompute_c1c=old['controller_order'][:4],mechanism_source_recompute_c2=extra[:4],bootstrap_seed=202610193201,bootstrap_replicates=5000,unseen_bootstrap='Fixed type strata, equal type weights; sample controllers within each type, then paired states; both seeds retained',cpu_limit=18,gpu_limit=2,initial_training_missing_estimate=52416,initial_new_eval_estimate=55296,implementation_error_rerun_cap=8000,conservative_method_included=True,exact_caps_set_after_official_preflight=True,legacy_c1c_dispatch='Original nine learned types use corresponding held-type model; all others including conservative IPPO and three unseen types use frozen all-nine-type model',gradient_interpolation='Three nodes at0,(H-1)/2,H-1 linearly interpolated at each integer planning step; explicit prompt takes precedence over inaccurate reference to original piecewise-constant CEM',gradient_warm_start='Shift optimized full trajectory one step, repeat last action, interpolate back to3 nodes; reset Adam moments per physical MPC optimization',cbs_constraints='Negative own directed edge at conflict time; swept-disc conflict predicate; branch once per involved robot; hold goals; minimum sum-of-quanta CT; root counts toward node cap, cap1 returns independent shortest paths',cbs_grid='Same underlying FastGrid geometry, moves, tie order and goal append; continuous tracking inherited from PrioritizedAStar; no unrequested path smoothing',lock_interpretation='Pending user clarification; no unseen capability/probe correction may run before clarification receipt')
    write(HERE/'OPENING_CONFIG.json',config)
    seal('OPENING_CONFIG.receipt.json',[HERE/n for n in ('OPENING_CONFIG.json','TRANSFORMER_TRAINING_CONFIGS.json','UNSEEN_CONTROLLER_CONFIGS.json','BASE_PROTOCOL.json','TRAINING_STATES.json','setup_design.py')],before_any_new_training_or_rollout=True)
    print('Frozen104 historical controllers/16 training types;18 unseen configurations; regression subset60 controllers')
if __name__=='__main__':main()
