#!/usr/bin/env python3
"""Read-only DB search for fixed-fail/adaptive-success with matched controller UID."""
import csv,json,sqlite3
from collections import defaultdict
from pathlib import Path
ROOT=Path('/home/zhihan/research/Basin_C1');OUT=Path(__file__).resolve().parent
C=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True)
C.row_factory=sqlite3.Row
f=next(r for r in csv.DictReader((ROOT/'diagnostics/orthoflow3_db_shared_mode_transfer_v1/train_mode_counts.csv').open()) if r['mode_id']=='0')
eta=C.execute('SELECT eta_uid FROM eta WHERE eta1=? AND eta2=? AND eta3=?',tuple(float(f[k]) for k in ('eta1','eta2','eta3'))).fetchone()[0]
by=defaultdict(lambda:defaultdict(dict)); group={}
for r in C.execute('''SELECT r.state_uid,r.controller_uid,r.eta_uid,r.seed_key,r.success,s.source_group,s.identity_quality
 FROM rollout r JOIN state s USING(state_uid) JOIN scenario sc USING(scenario_uid)
 JOIN controller_config cc USING(controller_uid)
 WHERE sc.name='DoubleBottleneck_4A' AND r.compatibility_quality='EXACT_REUSE'
 AND cc.orthoflow3_sha256='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38'
 AND cc.safety_config_hash='certified_hard_projection_v1'
 AND cc.conditioning_version='true_t0_latched_eta_v1'
 AND r.conflict_quarantined=0 AND r.numerical_failure=0'''):
    key=(r['state_uid'],r['controller_uid']);group[key]=(r['source_group'],r['identity_quality'])
    if r['seed_key'].startswith('{"future_index":'):
        by[key][r['eta_uid']][r['seed_key']]=int(r['success'])
rows=[]; unresolved=[];certified_partial=[]; inspected=0; fixedtested=0; fixed_b15=0;fixed_fail=0;robustother=0
for (state,ctl),etas in by.items():
    if eta in etas:inspected+=1
    fixedkeys=etas.get(eta,{})
    standard=[json.dumps({'future_index':i},separators=(',',':')) for i in range(16)]
    robust=[]
    for e,ss in etas.items():
        if e==eta:continue
        if all(k in ss for k in standard) and sum(ss[k] for k in standard)>=15:robust.append(e)
    if robust:robustother+=1
    missing=[k for k in standard if k not in fixedkeys]
    if missing:
        if robust:
            existing_success=sum(fixedkeys.values())
            rec={'state_uid':state,'controller_uid':ctl,'source_group':group[(state,ctl)][0],
                 'fixed_existing_standard_seeds':16-len(missing),'fixed_existing_successes':existing_success,
                 'missing_standard_seeds':len(missing),'other_B15_eta_count':len(robust)}
            if existing_success>=15:certified_partial.append(rec)
            elif existing_success+len(missing)>=15:unresolved.append(rec)
        continue
    fixedtested+=1
    fk=sum(fixedkeys[k] for k in standard)
    fixed_b15+=int(fk>=15);fixed_fail+=int(fk<15)
    if fk<15 and robust:
        rows.append({'state_uid':state,'controller_uid':ctl,'source_group':group[(state,ctl)][0],
                     'fixed_successes16':fk,'robust_other_eta_count':len(robust),
                     'robust_other_eta_uids':';'.join(sorted(robust))})
with (OUT/'db_fixed_fail_candidates.csv').open('w',newline='') as fp:
    w=csv.DictWriter(fp,fieldnames=['state_uid','controller_uid','source_group','fixed_successes16','robust_other_eta_count','robust_other_eta_uids']);w.writeheader();w.writerows(rows)
with (OUT/'db_fixed_fail_unresolved.csv').open('w',newline='') as fp:
    w=csv.DictWriter(fp,fieldnames=['state_uid','controller_uid','source_group','fixed_existing_standard_seeds','fixed_existing_successes','missing_standard_seeds','other_B15_eta_count']);w.writeheader();w.writerows(unresolved)
summary={'state_controller_profiles_with_any_fixed_seed':inspected,
         'state_controller_profiles_with_fixed_standard16':fixedtested,
         'fixed_standard16_B15_profiles':fixed_b15,'fixed_standard16_fail_profiles':fixed_fail,
         'profiles_with_other_B15_eta':robustother,'fixed_fail_adaptive_success_profiles':len(rows),
         'partial_fixed_certified_B15_profiles':len(certified_partial),
         'other_B15_but_fixed_unresolved_profiles':len(unresolved),
         'minimum_fixed_continuations_to_resolve_unresolved':sum(x['missing_standard_seeds'] for x in unresolved),
         'verdict':'DB_FIXED_FAIL_COHORT_NOT_AVAILABLE' if not rows else 'FOUND'}
(OUT/'db_fixed_fail_search.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
