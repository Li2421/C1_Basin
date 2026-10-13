"""Record the isolated-target falsification without a deployment claim."""
import csv
from .study import HERE,BASE,read,write,sha


def run():
    assert len(list((HERE/'fits').glob('*/*/seed*/complete.json')))==30
    d=read(HERE/'matrix_results.json');mech=read(HERE/'interaction_results.json');rows=[]
    for r in d['rows']:
        if r['reference']!='flow' or r['panel']!=('validation' if r['fold']=='joint' else 'all_held'):continue
        c={x['comparison']:x for x in d['comparisons'] if all(x[k]==r[k] for k in ('fold','scene','panel','reference','condition'))}
        row={k:r[k] for k in ('fold','scene','condition','N')}
        row.update(success_lo=r['selected_success_bounds'][0],success_hi=r['selected_success_bounds'][1],state_eta_corr=r['state_eta_corr_midpoint_sensitivity'],collisions=r['selected_collision'])
        for short,long in [('mismatch','matched_minus_same_scene_wrong_state'),('fixed','selection_minus_posthoc_best_fixed_per_controller')]:
            row[short+'_CI_lo'],row[short+'_CI_hi']=c[long]['censor_CI95']
        rows.append(row)
    with (HERE/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['Stage18 — direct interaction target versus full-Q regression','',
        'COMPLETE: 30 fits and all four held folds, Q16 unseen-controller panels, primary S x eta / S x C x eta bootstrap.',
        'New rollouts = 0; queries = 0. Dataset, physical/controller inputs, architecture, draws, optimization and source-only normalization unchanged between the two new arms.',
        'Primary source-VAL checkpoint rule is interaction error for both new arms. BCE models are reused references, not a pure loss-only causal contrast because their checkpoint rule differs.',
        'The interaction arm removes source-TRAIN state and eta additive means separately within every scene/controller. Unknown outcomes remain censored. Its secondary selection score adds an existing frozen prior and is uncalibrated. Do not label this score a validated success probability.','',
        'Primary raw interaction prediction versus zero-interaction, Flow-reference controllers:']
    for r in mech['rows']:
        if r['reference']=='flow' and r['first']=='interaction_mse' and r['second']=='zero':
            lines.append(f"{r['fold']} {r['scene']} {r['interaction']}: {r['skill_identification_bounds']}; physical-state/censor CI {r['censor_CI95']}")
    lines+=['','Outcome:',
        'Both objectives learn useful Toy source/joint interactions. Neither produces reliable positive S x eta or S x C x eta skill in any held scene.',
        'Every held-scene correct-state-minus-mismatch interval for the interaction selector includes zero. The Ring score improves over full-Q regression globally, but still loses to a per-controller fixed policy; it is not adaptive transfer.',
        'The independent opened K16/Q16 unseen-controller panels also fail the state-use test. All panels remain DEVELOPMENT.',
        'Therefore mean-effect competition is not a sufficient explanation or repair. Do not launch a loss-weight, learning-rate, width, or depth sweep.',
        'A separate audit found an important coverage/optimization confound: Stage17 native Ring gets only 6000*120/19/8192 = 4.626 expected draws per state-eta cell, versus 12000*128/8192 = 187.5 in the earlier successful single-controller fit. Its native TRAIN branch is underfit, even though other smaller controller groups overfit.',
        'These exposure counts are exact expectations, not proof of causation. Old/current models also differ in controller feature channels, so a matched-input, matched-draw causal control is required before interpreting the gap as interference or insufficient optimization.',
        'Next: audit that native branch with fixed current input/network, matched native draws versus a dense exposure control, and a same-shape zero-controller-feature control. No new simulation and no LOSO success claim from this diagnostic.','',
        'Artifacts: protocol.json, target_audit.json, training_targets.npz, fits/, matrix_results.json, interaction_results.json, q16_*_results.json, raw predictions, saved bootstrap replicates, summary.csv.']
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    write(HERE/'completion.json',dict(status='COMPLETE_DIAGNOSTIC_NO_TRANSFER_REPAIR',new_rollouts=0,fits=30,
        root_cause_closed=False,mean_effect_competition_sufficient=False,
        files={name:sha(HERE/name) for name in ('matrix_results.json','interaction_results.json','q16_toy_give_way_results.json','q16_ring_exchange_results.json')}))


if __name__=='__main__':run()
