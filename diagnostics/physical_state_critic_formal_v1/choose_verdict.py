"""Apply the predeclared conservative A–E verdict rule to all ten panels."""
from __future__ import annotations

import json
from pathlib import Path

from diagnostics.physical_state_critic_formal_v1 import consolidate

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def main():
    plan=read(HERE/'unified_train_plan.json')
    assert 'verdict_rule' in plan
    rows={p:read(HERE/'evaluation'/p/'metrics.json') for p in consolidate.PANELS}
    assert len(rows)==10 and all(v['test_states']==128 for v in rows.values())

    def counts(names,other):
        plus=minus=0
        for name in names:
            row=rows[name]['paired_comparisons'][f'Q_state_minus_{other}']['B15_confirmed']
            plus+=int(row['ci95'][0]>0)
            minus+=int(row['ci95'][1]<0)
        return dict(supported_gains=plus,supported_losses=minus,total=len(names))

    within=[p for p in rows if p.startswith('within_')]
    loso=[p for p in rows if p.startswith('loso_')]
    assert len(within)==6 and len(loso)==4
    within_eta=counts(within,'Q_eta')
    within_safety=counts(within,'Safety')
    loso_eta=counts(loso,'Q_eta')
    within_ok=(within_eta['supported_gains']>=4 and within_safety['supported_gains']>=4)
    loso_ok=(loso_eta['supported_gains']>=2 and loso_eta['supported_losses']==0)
    fixed=[]
    for name,row in rows.items():
        if row['diagnostics']['state_effect_correlation'] is None:continue
        z=row['diagnostics']
        a=z['local_pairwise_state']['accuracy'];b=z['local_pairwise_eta_only']['accuracy']
        fixed.append(dict(panel=name,state_effect_correlation=z['state_effect_correlation'],
                          pairwise_state=a,pairwise_eta_only=b,
                          learned=bool(z['state_effect_correlation']>.25 and a is not None and b is not None and a>b)))
    signal=any(r['learned'] for r in fixed) or within_eta['supported_gains']>0 or loso_eta['supported_gains']>0
    if within_ok and loso_ok:
        verdict='STATE_CONDITIONING_CONSISTENTLY_VALIDATED'
    elif within_ok:
        verdict='WITHIN_SCENE_VALIDATED_LOSO_MIXED'
    elif loso_ok:
        verdict='LOSO_VALIDATED_WITHIN_SCENE_MIXED'
    elif signal:
        verdict='STATE_SIGNAL_REAL_BUT_SELECTION_GAIN_LIMITED'
    else:
        verdict='GENERALIZATION_FAILURE_REMAINS'
    rationale=(f"Predeclared paired B15 rule: within-scene state–eta supported gains "
               f"{within_eta['supported_gains']}/6, state–Safety gains "
               f"{within_safety['supported_gains']}/6; FF LOSO state–eta gains "
               f"{loso_eta['supported_gains']}/4 and losses {loso_eta['supported_losses']}/4. "
               f"Fixed-bank learned-state diagnostic present: {signal}. "
               f"All ten panels use 128 fresh physical TEST states and report numerical bounds separately.")
    output=dict(verdict=verdict,rationale=rationale,within_eta=within_eta,
                within_safety=within_safety,loso_eta=loso_eta,fixed_bank_diagnostics=fixed,
                rule=plan['verdict_rule'],all_panels_complete=True)
    (HERE/'verdict.json').write_text(json.dumps(output,indent=2)+'\n')
    consolidate.main(verdict,rationale)
    print(json.dumps(output))


if __name__=='__main__':main()
