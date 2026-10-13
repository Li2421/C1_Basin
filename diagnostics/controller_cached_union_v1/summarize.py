"""Compact, scope-explicit Stage17 record; no new fits or rollouts."""
import csv
from .study import HERE, read, write, sha


def run():
    d=read(HERE/'matrix_results.json'); rows=[]
    for r in d['rows']:
        if r['reference']!='flow': continue
        if r['panel']!=('validation' if r['fold']=='joint' else 'all_held'): continue
        key={k:r[k] for k in ('fold','scene','panel','reference','condition')}
        comp={a['comparison']:a for a in d['comparisons'] if all(a[k]==v for k,v in key.items())}
        m=next(a for a in d['mechanisms'] if all(a[k]==v for k,v in key.items()))
        a=dict(**key,N=r['N'],success_lo=r['selected_success_bounds'][0],success_hi=r['selected_success_bounds'][1],
            Q_MSE=r['Q_MSE_midpoint_sensitivity'],state_eta_corr=r['state_eta_corr_midpoint_sensitivity'],
            collisions=r['selected_collision'],unknown=r['selected_numerical_unknown'],
            oracle_lo=r['oracle_success_bounds'][0],oracle_hi=r['oracle_success_bounds'][1],
            reversal_pairs=m['reversal']['qualifying_pairs'],confirmed_pairs=m['reversal']['independently_confirmed_pairs'])
        for short,long in [('mismatch','matched_minus_same_scene_wrong_state'),('best_fixed','selection_minus_posthoc_best_fixed_per_controller'),('eta','selection_minus_eta_only')]:
            v=comp[long]
            for label,pair in [('gain',v['identification_bounds']),('CI',v['censor_CI95'])]:
                a[short+'_'+label+'_lo'],a[short+'_'+label+'_hi']=pair
        rows.append(a)
    with (HERE/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['Stage17 — compatible cache union + state-erased supervision control','',
        'COMPLETE: 60 fits, all four held-scene panels, independent opened Q16 panels, and controller interaction audit.',
        '97,280 cached outcomes; 784 physical states; 4,032 state-controller rows; 1,794 numerical unknowns. New rollouts: 0.',
        'The union is heterogeneous: native Ring has 512 TRAIN states, while the crossed goal/Flow operator panel still has 32 TRAIN states per scene. A larger total is not 512 states for every controller.',
        'SOURCE-only gradient labels, normalization and checkpoint choice. No physical TRAIN/VAL overlap. All target panels are DEVELOPMENT, not final confirmation.',
        'K4/Q2-Q4 factorial panels are mechanism diagnostics; the K16/Q16 panel contains 32 states per scene and two previously unseen analytic controllers. Do not report the former as B15 benchmark success.','',
        'Flow-reference operator panel: success and mismatch/fixed-policy comparisons (probability units):',
        'fold scene model N success_lower success_upper matched-minus-mismatch_95CI model-minus-bootstrap-best-per-C-fixed_95CI']
    for r in rows:
        if r['condition'] not in ('eta_only','union_native_slots','union_slots_state_erased'):continue
        lines.append(f"{r['fold']} {r['scene']} {r['condition']} {r['N']} {r['success_lo']:.5f} {r['success_hi']:.5f} [{r['mismatch_CI_lo']:.5f},{r['mismatch_CI_hi']:.5f}] [{r['best_fixed_CI_lo']:.5f},{r['best_fixed_CI_hi']:.5f}]")
    lines+=['','State-dependent controller x eta skill after removal of all C/eta means:',
        'Positive skill means lower squared error than predicting zero interaction; censor bounds and bootstrap recenter inside each physical-state replicate.']
    for r in read(HERE/'interaction_results.json')['rows']:
        if r['reference']=='flow' and r['interaction']=='state_controller_eta' and r['first']=='union_native_slots' and r['second']=='zero':
            lines.append(f"{r['fold']} {r['scene']}: skill={r['skill_identification_bounds']}, CI={r['censor_CI95']}")
    lines+=['','Interpretation:',
        'Toy joint/source VAL genuinely carries state/controller/eta interaction; it is checkpoint-selection data, not a final confirmation.',
        'No held scene shows reliable positive controller x state x eta skill. DB/Four/Ring skill is significantly negative relative to zero; Toy interval includes zero.',
        'Held-Toy has a tiny positive K4 matched-minus-mismatch difference, but remains far below best per-controller fixed eta; this is not successful adaptive transfer.',
        'The independent opened Q16 unseen-controller panels do not repair this conclusion: all matched-minus-mismatch CIs include zero, and no strong discovery reversal pair qualifies.',
        'Therefore combining this existing support is not a sufficient repair. This does NOT prove that every useful data expansion would fail, nor that Q is theoretically unlearnable.',
        'Safety: no collision in these retained panels. Numerical unknowns remain bounded, not silently counted as success/failure. There is no compatible matched Safety baseline for every diagnostic controller, so none is fabricated.',
        'Next narrow test: learn the state x eta interaction directly, with a same-network full-Q regression control. This tests interference from nuisance mean effects; it is not yet a causal root-cause claim.','',
        'Artifacts: protocol.json; dataset_audit.json; dataset.npz; rows.json; fits/; matrix_results.json; interaction_results.json; q16_*_results.json; corresponding raw predictions and bootstrap NPZs; summary.csv.']
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    write(HERE/'completion.json',dict(status='COMPLETE_DEVELOPMENT_DIAGNOSTIC_TRANSFER_NOT_REPAIRED',fits=60,new_rollouts=0,
        files={name:sha(HERE/name) for name in ('matrix_results.json','interaction_results.json','q16_toy_give_way_results.json','q16_ring_exchange_results.json')},
        fundamental_cause_resolved=False))


if __name__=='__main__':run()
