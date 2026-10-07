"""Pre-registered critic gate; freeze only if train/dev protections pass."""
from datetime import datetime,timezone
from pathlib import Path
import json
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a


def main():
    if (a.OUT/'phase_a/critic_frozen.json').exists():
        raise RuntimeError('Critic already frozen; no automatic reselection')
    protected={**a.load(a.OUT/'canonical_hashes_before.json'),**a.load(a.OUT/'runtime_artifacts.json')['files']}
    for name,value in protected.items():
        if a.sha(a.ROOT/name)!=value:raise RuntimeError(f'Canonical artifact changed: {name}')
    candidates=[]
    for seed in a.learn.SEEDS:
        train=a.load(a.OUT/f'phase_a/critic/seed{seed}/training.json')
        metrics=a.load(a.OUT/f'phase_a/critic/seed{seed}/metrics.json')
        assert a.sha(Path(train['checkpoint']))==train['sha256']
        summary=metrics['summary'];gates={}
        for sc in a.SCENARIOS:
            s=summary[sc]
            gates[sc+'_unresolved_not_increased']=s['unresolved']<=s['old_unresolved']
            if sc!='ring_exchange':
                gates[sc+'_robust_drop_le_005']=(s['old_robust']-s['selected_robust'])/s['states']<=.05
                gates[sc+'_exploitation_negligible']=s['exploitation']<=max(1,s['old_exploitation'])
        r=summary['ring_exchange']
        gates['ring_misses_improve_or_ceiling']=(r['miss']<r['old_misses']) if r['old_misses'] else r['miss']==0
        gates['ring_positive_evidence_not_unknown_gain']=(r['selected_robust']>r['old_robust']) if r['old_misses'] else r['selected_robust']>=r['old_robust']
        gates['ring_exploitation_improve_or_zero']=(r['exploitation']<r['old_exploitation']) if r['old_exploitation'] else r['exploitation']==0
        strata={}
        for sc in a.SCENARIOS:
            for pop in ('v2','initial_expansion'):
                rr=[r for r in metrics['states'] if r['scenario']==sc and r['population']==pop]
                if rr:strata[sc+':'+pop]={'states':len(rr),'old_robust':sum(r['old_robust'] is True for r in rr),
                                        'new_robust':sum(r['selected_robust'] is True for r in rr),
                                        'oracle':sum(r['oracle_robust'] is True for r in rr)}
        candidates.append({'training':train,'gates':gates,'passed':all(gates.values()),'summary':summary,'strata':strata})
    passing=[r for r in candidates if r['passed']]
    a.dump('phase_a/development_review.json',{'candidates':candidates,'acceptance_protocol_sha256':a.sha(a.OUT/'phase_a/acceptance_protocol.json'),
                                             'fresh_confirmation_inspected':False,'passed_candidates':len(passing)})
    if not passing:
        print('No candidate passed. Train/dev diagnosis required; no fresh cohort created.',flush=True)
        return 2
    chosen=min(passing,key=lambda r:(r['training']['validation_loss'],r['training']['seed']))
    t=chosen['training']
    result={'development_frozen':True,'created_at':datetime.now(timezone.utc).isoformat(),
            'checkpoint':t['checkpoint'],'checkpoint_sha256':t['sha256'],'seed':t['seed'],
            'generator_checkpoint':str(a.frozen.GENERATOR_CKPT),'generator_sha256':a.sha(a.frozen.GENERATOR_CKPT),
            'normalization':str(a.frozen.NORMALIZATION),'normalization_sha256':a.sha(a.frozen.NORMALIZATION),
            'K_stochastic':16,'candidate_set':'frozen mean +16stochastic,noothercandidates','mean_only_deployment':False,
            'basis_changed':False,'generator_retrained':False,'fresh_outcomes_seen':False,
            'development_summary':chosen['summary'],'development_gates':chosen['gates'],
            'data_manifests':[{'path':str(a.OUT/p),'sha256':a.sha(a.OUT/p)} for p in
                  ['phase_a/proposals.json','phase_a/proposal_Q_evidence.json','initial_expansion/phase_a/proposals.json',
                   'initial_expansion/phase_a/proposal_Q_evidence.json','phase_a/critic/config.json']]}
    a.dump('phase_a/critic_frozen.json',result)
    lines=['# Phase A development freeze','',f"Selected canonical critic seed {t['seed']}; generator and original representation unchanged.",
           '', 'Fresh confirmation has NOT yet been evaluated. This is not the Phase-A final decision.', '',
           '| Scenario | States | Oracle B15 | Old critic B15 | New critic B15 | Old/new misses | Old/new exploitation |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for sc,s in chosen['summary'].items():
        lines.append(f"| {sc} | {s['states']} | {s['oracle_robust']} | {s['old_robust']} | {s['selected_robust']} | {s['old_misses']}/{s['miss']} | {s['old_exploitation']}/{s['exploitation']} |")
    (a.OUT/'phase_a/DEVELOPMENT_FREEZE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
