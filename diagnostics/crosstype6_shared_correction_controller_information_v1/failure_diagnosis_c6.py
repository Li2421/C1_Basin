"""Observed failure tags and ablation evidence, without unsupported causal claims."""
from common import *
from menu_c6 import METHODS
from statistics_c6 import LEARNING
from catalog_names import full_type

def main():
    if (HERE/'FAILURE_DIAGNOSIS.receipt.json').exists():return
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    names=list(d['names']);types=d['types'];label=d['labels'];steps=d['trigger_steps'];no_repair=np.zeros((122,24,2),bool);requery=np.zeros_like(no_repair)
    for r in read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states']:
        ci=names.index(r['controller']);si=r['state_index']
        for branch,mi in enumerate((1,2)):
            a=r['methods'][METHODS[mi]];no_repair[ci,si,branch]=not np.any(a['eta_a']) and not np.any(a['eta_b']) and a.get('late_source','given')=='given';requery[ci,si,branch]=a.get('late_source','given')=='requery'
    r=read(HERE/'T1_T7_RESULTS.json');output={}
    for t in sorted(LEARNING):
        mask=types==t;row=dict(type=full_type(t),branches={},ablations={k:r['groups'][t]['comparisons'][k] for k in ('T2_online','T3_offline','T4_threshold','T5_state_trigger','T7_vs_C5')},selected_thresholds=r['selected_threshold_counts'][t])
        for branch,mi in enumerate((1,2)):
            ll=label[mask,:,mi];failed=ll==0;nr=no_repair[mask,:,branch,None];ts=steps[mask,:,mi];first=label[mask,:,0];human=label[mask,:,6];allfive=label[mask,:,:5];known_five_fail=np.all(allfive==0,axis=2)
            row['branches'][METHODS[mi]]=dict(success=r['groups'][t]['methods'][METHODS[mi]],point_at_least90=bool(r['groups'][t]['methods'][METHODS[mi]]['point']>=90),known_failed_controller_state_seed_cases=int(failed.sum()),unknown_cases=int(np.sum(ll<0)),selected_no_repair_known_failures=int(np.sum(failed&nr)),nonzero_schedule_never_triggered_known_failures=int(np.sum(failed&~nr&(ts<0))),trigger_at_or_after200_known_failures=int(np.sum(failed&(ts>=200))),trigger_before200_known_failures=int(np.sum(failed&(ts>=0)&(ts<200))),first_layer_success_online_layer_known_failure=int(np.sum((first==1)&failed)),human_success_all_five_learned_choices_known_failure=int(np.sum((human==1)&known_five_fail)),chosen_requery_fraction=float(requery[mask,:,branch].mean()),chosen_no_repair_fraction=float(no_repair[mask,:,branch].mean()))
        row['interpretation_limits']=['Trigger-time tags describe observed failures, not same-eta counterfactual effects; layers may select different eta pairs and thresholds.','Threshold/global-trigger ablations are the prescribed method comparisons, not causal isolation with every other action held fixed.','Human succeeds while every evaluated learned choice fails identifies a measured pipeline gap; full proposal coverage is unmeasured, so generator coverage versus critic choice cannot be separated.','Timeout remains timeout; it is not relabeled deadlock.']
        output[full_type(t)]=row
    write(HERE/'FAILURE_DIAGNOSIS.json',dict(utc=utc(),types=output,evidence='DEVELOPMENT',no_extra_rollouts=True,no_unseen_final_test=True))
    seal('FAILURE_DIAGNOSIS.receipt.json',[HERE/'FAILURE_DIAGNOSIS.json',HERE/'failure_diagnosis_c6.py',HERE/'NEW_EVAL_BUNDLE.receipt.json',HERE/'GROUP_MEMBERSHIP.json'],observational_not_overclaimed_causal_attribution=True)

if __name__=='__main__':main()
