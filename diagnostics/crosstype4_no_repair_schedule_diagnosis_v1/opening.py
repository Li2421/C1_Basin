"""Fixed controller/state draws and previous frozen decisions; no reselection."""
from common import *
import csv
def main():
    prereg();base=read(HERE/'BASE_PROTOCOL.json');c3=D/'crosstype3_unseen_controller_schedule_v1';c1=D/'crosstype1c_two_eta_schedule_v1'
    with (c1/'ZOO_MANIFEST.csv').open() as f:flow=[r['controller'] for r in csv.DictReader(f) if r['type']=='Flow']
    assert len(flow)==14
    names=base['controller_order'];a1=[n for n in names if base['types'][n] in ('transformer_bc','conflict_based_search')];a2=[n for n in names if base['types'][n] in ('centralized_mppi','distributed_cross_entropy')];coverage=a1+a2+flow[:6];assert len(a1)==12 and len(a2)==12 and len(coverage)==30
    new=read(c3/'NEW_EVAL_CHOSEN_ACTIONS.json');old=read(c1/'NEW_EVAL_CHOSEN_ACTIONS.json');A=[];states={};candidate_cache={root:read(root/'CANDIDATES.json')['states'][48:] for root in (c3,c1)}
    from frozen_networks import stable_int
    for root,wanted in [(c3,a1+a2),(c1,flow)]:
        source=new if root==c3 else old;ss=read(root/'NEW_EVAL_STATES.json');lookup={(r['controller'],r['state_index']):r for r in source['states']}
        for name in wanted:
            count=48 if name in a1 else 24;selected=sorted(np.random.default_rng(stable_int('crosstype4-A-coverage',202610194101,name)).choice(count,8,replace=False).tolist()) if name in coverage else []
            for si,s in enumerate(ss[:count]):
                r=lookup[name,si];action=r['methods']['self_schedule'];states[s['state_uid']]=s;proposals=candidate_cache[root][si]['proposals'];assert len(proposals)==8
                A.append(dict(controller=name,type=base['types'][name],state_uid=s['state_uid'],source_state_index=si,state=s,source=str(root),source_choice_receipt=str(root/'NEW_EVAL_CHOICES.receipt.json'),selected_action=action,proposals=proposals,coverage=si in selected,diagonal=bool(np.array_equal(action['eta_a'],action['eta_b']))))
    b_eval=[n for n in names if base['types'][n] in ('Flow','ring_ippo_conservative','distributed_shooting','centralized_sampling','cross_entropy','distributed_cross_entropy','centralized_mppi','prioritized_astar','transformer_bc','gradient_mpc','conflict_based_search')];assert len(b_eval)==74
    write(HERE/'A_OBJECTS.json',dict(utc=utc(),records=A,states=list(states.values()),controller_order=a1+a2+flow,coverage_controllers=coverage,coverage_master_seed=202610194101,all_selected_actions_from_frozen_history=True))
    write(HERE/'OPENING_CONFIG.json',dict(utc=utc(),A1=a1,A2=a2,A3=flow,A_coverage_controllers=coverage,B_training=names,B_evaluation=b_eval,B_training_types=19,B_training_controllers=122,B_eval_types=11,B_eval_controllers=74,B_new_states=24,B_new_state_seed_start=202610194301,B_new_state_seed_stop_inclusive=202610194324,B_new_namespace='crosstype4_no_repair_new_evaluation',extra_training_seed=202610194201,early_equivalence_seed=202610194401,A_bootstrap_seed=202610194501,B_bootstrap_seed=202610194502,replicates=5000,source_sets=191,three_seed_branch_fits=382,cpu_limit=18,gpu_limit=2,evidence='DEVELOPMENT',only_allowed_main_role='本轮主方法 = 学习双 η 含不修正·可选按进度'))
    seal('OPENING_CONFIG.receipt.json',[HERE/'OPENING_CONFIG.json',HERE/'A_OBJECTS.json',HERE/'BASE_PROTOCOL.json',HERE/'TRAINING_STATES.json',HERE/'opening.py',HERE/'PREREGISTRATION.receipt.json'],before_A_or_B_labels=True,A_choices_not_reselected=True)
    print('Fixed A38 controllers/30 coverage; B122 training,74 evaluation controllers',flush=True)
if __name__=='__main__':main()
