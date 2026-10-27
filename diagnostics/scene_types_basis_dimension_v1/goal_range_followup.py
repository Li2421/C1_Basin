"""Budget-matched B3 range control, separate from the seven-family verdict.

Both32 pools contain the same original16 candidates. Six paired boundary
points and ten paired Sobol points differ only in the G coordinate.
"""
from dispatch import *
from discovery import cohorts
from acceleration import accelerated_case
from scipy.stats import qmc
import datetime

def build(protocol):
    frozen=read(HERE/'SCENES_FROZEN.json');cases=[];labels={}
    for name,pool in protocol['pools'].items():
        for s,c,i in cohorts(frozen,range(4,8)):
            for k,sp in enumerate(pool):
                cc=accelerated_case(s,c,i,sp,purpose='goal_range_followup_holdout')
                cases.append(cc);labels.setdefault(name,[]).append(dict(uid=cc['rollout_uid'],candidate=k,type_id=s['type_id'],controller=c['type_code'],state_index=i))
    return cases,labels

def freeze():
    path=HERE/'GOAL_RANGE_FROZEN.json'
    if path.exists():return read(path)
    families=read(HERE/'FAMILIES_FROZEN.json')['families'];anchor=next(f['candidate_pool']for f in families if set(f['columns'])=={'G','P','R'})
    face=read(HERE/'GOAL_ZERO_FACE_FROZEN.json')['pool'];assert len(anchor)==16 and len(face)==6
    narrow=[];wide=[];latent=qmc.Sobol(3,scramble=True,seed=2026105800).random_base2(4)[:10]
    for sp in face:
        assert sp['eta'][0]==0. and not sp.get('coefficients')
        wide.append(sp);narrow.append(dict(eta=[.5,*sp['eta'][1:]],coefficients={}))
    for u in latent:
        p=-.5+float(u[1]);r=.75*float(u[2])
        narrow.append(dict(eta=[.5+.75*float(u[0]),p,r],coefficients={}))
        wide.append(dict(eta=[1.25*float(u[0]),p,r],coefficients={}))
    protocol=dict(freeze_time=datetime.datetime.now().astimezone().isoformat(),pools=dict(narrow32=anchor+narrow,wide32=anchor+wide),seed=2026105800,primary_family_freeze_sha256=sha(HERE/'FAMILIES_FROZEN.json'),boundary_control_sha256=sha(HERE/'GOAL_ZERO_FACE_FROZEN.json'),paired_topup_latents=latent,selection='Supplementary original-request parameter-range ablation. Frozen after primary test began but before executing new range candidates; no heldout outcomes inspected or used to choose pools. Primary seven families, fixed parameters and dimension verdict remain unchanged.',matched_rule='Original16 common. Six boundary pairs share P/R with G=0 versus0.5. Ten Sobol pairs share P/R and latent coordinate, G=.5+.75u versus1.25u. All32 requested in each pool, exact repeated requests reuse work.',evidence_gap='Old small range study did not test these new maps. Existing G=0 K6 boundary control is unequal-budget and does not cover the widened interior; it cannot isolate the Goal-box confound.',limits='Finite structured candidate-pool comparison, not continuous optimization. Supplementary evaluation is not included in the primary dimension verdict.')
    cases,labels=build(protocol);primary={x['uid']for ll in read(HERE/'HOLDOUT_LABELS.json').values()for x in ll};extra={c['rollout_uid']for c in cases}-primary
    profiles=read(HERE/'PAPER_PARAMETERS_FROZEN.json');n_profiles=len(profiles['historical'])+sum(profiles['selected_development'][m]!=v for m,v in profiles['historical'].items());n_physical=len({(x['type_id'],x['state_index'])for x in labels['narrow32']})
    expected=len(primary|{c['rollout_uid']for c in cases})+n_profiles*n_physical
    assert expected<=15700,(expected,'heldout unique-work budget')
    protocol.update(primary_unique_requests=len(primary),followup_unique_requests=len({c['rollout_uid']for c in cases}),additional_vs_primary=len(extra),paper_profiles=n_profiles,expected_all_heldout_unique_requests=expected,heldout_budget=15700)
    dump(path,protocol);print('MATCHED RANGE CONTROL FROZEN',len(extra),'additional unique requests;',expected,'total heldout including papers',flush=True);return protocol

def main():
    protocol=read(HERE/'GOAL_RANGE_FROZEN.json');assert sha(HERE/'FAMILIES_FROZEN.json')==protocol['primary_family_freeze_sha256']
    cases,labels=build(protocol);p=HERE/'GOAL_RANGE_LABELS.json'
    if p.exists():assert digest(read(p))==digest(labels)
    else:dump(p,labels)
    return execute(cases,'goal_range_holdout')

if __name__=='__main__':
    import sys
    freeze()if'--freeze'in sys.argv else main()
