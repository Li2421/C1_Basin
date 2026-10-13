"""Read-only integrity assertions and machine-readable verdict; no model calls."""
import csv,json,xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def read(name):
    with (HERE/name).open(newline='') as f:return list(csv.DictReader(f))

def main():
    ring=read('ring_directional_mechanism.csv');toy=read('toy_yielding_mechanism.csv')
    traj=read('trajectory_geometry.csv');held=read('heldout_validation.csv')
    flow=read('flow_substep_decomposition.csv');summ=read('flow_substep_summary.csv')
    assert len(ring)==len(toy)==384 and len(traj)==len(held)==768
    assert len(flow)==240 and len(summ)==24
    assert Counter(r['b15_outcome'] for r in ring)==Counter(gain=48,loss=1,both_success=29,both_failure=306)
    assert Counter(r['b15_outcome'] for r in toy)==Counter(gain=10,loss=178,both_success=25,both_failure=171)
    assert all(int(float(r['noise_count']))==16 for r in ring+toy)
    assert max(float(r['zero_plus_correction_identity_error']) for r in ring+toy)<1e-12
    assert max(abs(float(r['goal_mean_difference'])-float(r['goal_pre_action_difference'])-
                   float(r['goal_projection_change_of_contrast'])) for r in toy)<1e-12
    assert max(abs(float(r['angular_abs_action_difference'])-
                   (float(r['new_pre_angular_abs'])-float(r['old_pre_angular_abs']))-
                   float(r['projection_angular_change_advantage'])) for r in ring)<1e-12
    assert max(max(float(r['old_cached_action_error']),float(r['new_cached_action_error'])) for r in summ)<1e-8
    max_violation=max(float(r['new_action_max_violation']) for r in summ)
    assert max_violation<1e-8
    for f in (HERE/'figures').glob('*.svg'):ET.parse(f)
    assert len(list((HERE/'figures').glob('*.svg')))==2
    metric=read('heldout_metric_summary.csv')
    assert len(metric)>0
    statuses={
      'GEOMETRIC_MECHANISM_SUPPORTED':{'status':'PARTIAL','ring':'terminal projection preserves absolute angular action on many Gain cases but no unique favorable signed direction; same effect widespread in non-gains',
          'toy':'flow feedback and later passage/event pattern supported; instantaneous pair clearance increases, so clearance-loss explanation rejected'},
      'LOCAL_THEORY_VALIDATED':{'status':'PARTIAL','scope':'cached J_eta and M B support fixed-state local directional readout; K/S only prior regular small-sample checks; finite Q16 eta is not local'},
      'HELDOUT_PREDICTION_VALIDATED':{'status':'FAIL','reason':'no geometry score exceeds discovery-label eta identity AP on grouped heldout; historical outcome leakage prevents prospective claim'},
      'CAUSAL_MEDIATION_SUPPORTED':{'status':'INSUFFICIENT_EVIDENCE','reason':'paired placement total effects but no safe full-loop intervention isolating angular coordination or yielding mediator'},
      'INSUFFICIENT_EVIDENCE':{'status':True,'scope':'unique beneficial Ring direction, prospective B15 condition, independent mediator effect, all-gain explanation'}
    }
    statuses['integrity']={
      'historical_full_evaluations':49152,'ring_candidates':len(ring),'toy_candidates':len(toy),
      'old_first_action_seed_evaluations_reused':12288,'paired_seed0_trajectories_reused':len(traj),
      'previously_completed_action_only_cases':len(summ),'previously_completed_field_integrations':48,
      'new_full_continuations':0,'new_short_continuations':0,'new_training':0,'gpu':0,
      'max_cached_first_action_reconstruction_error':max(max(float(r['old_cached_action_error']),float(r['new_cached_action_error'])) for r in summ),
      'max_action_safety_violation':max_violation,
      'ring_ambiguous_candidates':sum(float(r['ambiguous_agents'])>0 for r in ring),
      'ring_gain_shortest_finite_positive':sum(r['b15_outcome']=='gain' and float(r['shortest_action_difference'])>0 for r in ring),
      'ring_gain_projection_retention_positive':sum(r['b15_outcome']=='gain' and float(r['projection_angular_change_advantage'])>0 for r in ring),
      'ring_both_failure_projection_retention_positive':sum(r['b15_outcome']=='both_failure' and float(r['projection_angular_change_advantage'])>0 for r in ring),
      'toy_loss_goal_negative':sum(r['b15_outcome']=='loss' and float(r['goal_mean_difference'])<0 for r in toy),
      'toy_loss_clearance_positive':sum(r['b15_outcome']=='loss' and float(r['clearance_hdot_difference'])>0 for r in toy),
      'toy_loss_local_goal_positive':sum(r['b15_outcome']=='loss' and float(r['goal_local_differential'])>0 for r in toy)
    }
    (HERE/'verification.json').write_text(json.dumps(statuses,indent=2,ensure_ascii=False)+'\n')
    print('PASS: cache-only denominators, finite identities, action replay, safety, figures')

if __name__=='__main__':main()
