"""Aggregate the frozen common-cloud and directional-transfer audit.

This intentionally does not infer unqueried neighbor canonical eta values.
The only B63 evidence in this audit is the pre-frozen anchor canonical
evidence plus the resource-bounded 64-seed transfer at selected neighbors.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_local_basin_continuity_v1')

def jsonl(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

def mean(values): return float(np.mean(values)) if len(values) else float('nan')
def rank(values):
    order = np.argsort(values); result = np.empty(len(values), float)
    result[order] = np.arange(len(values), dtype=float)
    # tie-average ranks
    for val in np.unique(values):
        where = np.where(np.asarray(values) == val)[0]
        result[where] = np.mean(result[where])
    return result
def corr(a,b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0: return float('nan')
    return float(np.corrcoef(a,b)[0,1])

def main():
    anchors = json.loads((HERE/'anchor_manifest.json').read_text())['anchors']
    neighbors = json.loads((HERE/'neighbor_manifest.json').read_text())['neighbors']
    screen_plan = json.loads((HERE/'screen_plan.json').read_text())
    cross_plan = json.loads((HERE/'cross_transfer_plan_amended.json').read_text())
    plan_by_arm = {a['arm_id']:a for a in screen_plan['arms']}
    xplan_by_arm = {a['arm_id']:a for a in cross_plan['arms']}
    raw_all = []
    for directory in sorted((HERE/'raw').glob('screening*')):
        if directory.is_dir():
            for f in sorted(directory.glob('shard*.jsonl')): raw_all += jsonl(f)
    expected = sum(len(a['seeds']) for a in screen_plan['arms'])
    dedup = {}
    duplicate_screening = 0
    for row in raw_all:
        key=(row['arm_id'],row['seed'])
        if key in dedup:
            duplicate_screening += 1
            old=dedup[key]
            # Cancellation-race reruns must agree on the semantically relevant
            # deterministic tuple; otherwise do not silently choose one.
            if (old['success'],old['outcome'],old['terminal_step']) != (row['success'],row['outcome'],row['terminal_step']):
                raise RuntimeError(('nonidentical duplicate',key))
        else: dedup[key]=row
    raw=list(dedup.values())
    if len(raw) != expected:
        raise RuntimeError(f'incomplete unique screening {len(raw)} != {expected}')
    if any(r.get('execution_error') for r in raw): raise RuntimeError('screening execution errors')
    # one row per state / probe; the plan makes this key unambiguous
    grouped=defaultdict(list)
    for r in raw: grouped[(r['state_id'],r['probe_id'])].append(r)
    screen_rows=[]; q={}
    for (state,probe),rs in sorted(grouped.items()):
        if len(rs)!=8: raise RuntimeError((state,probe,len(rs)))
        arm=plan_by_arm[rs[0]['arm_id']]
        successes=sum(bool(r['success']) for r in rs)
        outcome=Counter(r['outcome'] for r in rs)
        row={'state_id':state,'probe_id':probe,'eta':json.dumps(arm['eta']),
             'eta_index':arm.get('eta_index'),'role':arm['role'],'anchor_rank':arm['anchor_rank'],
             'offset_steps':arm.get('offset_steps',0),'n':len(rs),'successes_8':successes,
             'Q_8':successes/8,'deadlock_8':outcome.get('strict_deadlock',0),
             'timeout_8':outcome.get('timeout',0),'collision_8':outcome.get('agent_collision',0)+outcome.get('wall_collision',0),
             'other_8':len(rs)-successes-outcome.get('strict_deadlock',0)-outcome.get('timeout',0)-outcome.get('agent_collision',0)-outcome.get('wall_collision',0),
             'mean_J_def':mean([float(r['J_def']) for r in rs]),'max_first_projection_retries':max(int(r['first_projection_retries']) for r in rs),
             'max_second_projection_retries':max(int(r['second_projection_retries']) for r in rs)}
        screen_rows.append(row); q[(state,probe)]=row
    write_csv(HERE/'screening_q_results.csv',screen_rows)

    distances={r['neighbor_id']:r for r in csv.DictReader((HERE/'state_pair_distances.csv').open())}
    # Common-cloud q-field metrics and screening basin sets.
    pair_q=[]; overlap=[]
    probes=[a['probe_id'] for a in screen_plan['arms'] if a['role']=='anchor' and a['anchor_rank']==0]
    for n in neighbors:
        a=anchors[int(n['anchor_rank'])]; aid=a['state_id']; bid=n['neighbor_id']
        qa=np.asarray([float(q[(aid,p)]['Q_8']) for p in probes]); qb=np.asarray([float(q[(bid,p)]['Q_8']) for p in probes])
        delta=np.abs(qa-qb); s8a={p for p,x in zip(probes,qa) if x==1}; s8b={p for p,x in zip(probes,qb) if x==1}
        s7a={p for p,x in zip(probes,qa) if x>=.875}; s7b={p for p,x in zip(probes,qb) if x>=.875}
        d=distances[bid]
        base={'anchor_rank':n['anchor_rank'],'anchor_state_id':aid,'neighbor_id':bid,'offset_steps':n['offset_steps'],
              'mean_abs_delta_Q8':mean(delta),'median_abs_delta_Q8':float(np.median(delta)), 'max_abs_delta_Q8':float(np.max(delta)),
              'pearson_q_field':corr(qa,qb),'spearman_q_field':corr(rank(qa),rank(qb)),
              'aggregate_position_displacement':d.get('aggregate_position_displacement'),'relative_geometry_change':d.get('relative_geometry_change'),
              'goal_error_change_l2':d.get('goal_error_change_l2'),'normalized_feature_distance':d.get('normalized_feature_distance'),
              'discrete_monitor_transition':d.get('discrete_monitor_transition'),
              'stuck_timer_change':d.get('stuck_timer_change')}
        pair_q.append(base)
        for label,x,y in [('S8',s8a,s8b),('S7',s7a,s7b)]:
            inter=len(x&y); union=len(x|y)
            overlap.append({'anchor_rank':n['anchor_rank'],'anchor_state_id':aid,'neighbor_id':bid,'offset_steps':n['offset_steps'],'screening_set':label,
                            'anchor_count':len(x),'neighbor_count':len(y),'intersection':inter,'union':union,
                            'jaccard':inter/union if union else float('nan'),'retention_anchor_to_neighbor':inter/len(x) if x else float('nan'),
                            'retention_neighbor_to_anchor':inter/len(y) if y else float('nan')})
    write_csv(HERE/'q_field_similarity.csv',pair_q); write_csv(HERE/'basin_overlap.csv',overlap)

    # B63 directional transfer. For zero targets use first eight screening
    # arms plus remaining 56 promotion arms; otherwise promotions provide 64.
    xraw=[]
    for directory in sorted((HERE/'raw').glob('cross64*')):
        if directory.is_dir():
            for f in sorted(directory.glob('shard*.jsonl')): xraw += jsonl(f)
    expectedx=sum(len(a['seeds']) for a in cross_plan['arms'])
    if len(xraw)!=expectedx: raise RuntimeError(f'incomplete cross {len(xraw)} != {expectedx}')
    if any(r.get('execution_error') for r in xraw): raise RuntimeError('cross execution errors')
    xgroup=defaultdict(list)
    for r in xraw: xgroup[r['arm_id']].append(r)
    promoted=[]; cross=[]
    amap={int(a['anchor_rank']):a for a in anchors}
    for armid,arm in sorted(xplan_by_arm.items()):
        rs=xgroup[armid]
        if arm['screen_seed_reuse_expected']:
            rs = rs + [r for r in grouped[(arm['state_id'],'ZERO')]]
        if len(rs)!=64: raise RuntimeError((armid,len(rs)))
        if len({r['seed'] for r in rs})!=64: raise RuntimeError(('duplicate seed',armid))
        suc=sum(bool(r['success']) for r in rs); out=Counter(r['outcome'] for r in rs)
        rec={'arm_id':armid,'anchor_rank':arm['anchor_rank'],'anchor_state_id':arm['anchor_state_id'],'neighbor_id':arm['state_id'],'offset_steps':arm['offset_steps'],
             'eta':json.dumps(arm['eta']),'n':64,'successes_64':suc,'Q_64':suc/64,'B63_retained':suc>=63,
             'deadlock_64':out.get('strict_deadlock',0),'timeout_64':out.get('timeout',0),'collision_64':out.get('agent_collision',0)+out.get('wall_collision',0),
             'mean_J_def':mean([float(r['J_def']) for r in rs]),'source_anchor_B63':'prior migration B63 canonical evidence',
             'direction':'anchor_eta_to_neighbor'}
        promoted.append(rec); cross.append(rec)
    write_csv(HERE/'promoted_64seed_results.csv',promoted); write_csv(HERE/'cross_transfer.csv',cross)

    # Set distances are deliberately marked unresolved: shared cloud screening
    # is not 64-seed robust evidence and neighbors were not oracle-labelled.
    setrows=[]
    for row in pair_q:
        setrows.append({'anchor_rank':row['anchor_rank'],'anchor_state_id':row['anchor_state_id'],'neighbor_id':row['neighbor_id'],'offset_steps':row['offset_steps'],
                        'status':'UNDERRESOLVED','reason':'common cloud is 8-seed screening only; no independently sampled B63 neighbor set'})
    write_csv(HERE/'sampled_set_distances.csv',setrows)

    # Canonical neighbor targets were intentionally not searched in this audit.
    canrows=[]
    for row in pair_q:
        canrows.append({'anchor_rank':row['anchor_rank'],'anchor_state_id':row['anchor_state_id'],'neighbor_id':row['neighbor_id'],'offset_steps':row['offset_steps'],
                        'canonical_neighbor_eta_available':False,'status':'NOT_EVALUABLE','reason':'no neighbor oracle/canonical selection run by this model-free continuity design'})
    write_csv(HERE/'canonical_vs_basin_jump.csv',canrows)

    zero=[]
    xbyneighbor={r['neighbor_id']:r for r in cross}
    for n in neighbors:
        aid=anchors[int(n['anchor_rank'])]['state_id']; bid=n['neighbor_id']; az=q[(aid,'ZERO')]; bz=q[(bid,'ZERO')]
        x=xbyneighbor.get(bid)
        zero.append({'anchor_rank':n['anchor_rank'],'anchor_state_id':aid,'neighbor_id':bid,'offset_steps':n['offset_steps'],
                     'anchor_zero_Q8':az['Q_8'],'neighbor_zero_Q8':bz['Q_8'],'zero_cross_promoted':bool(x and json.loads(x['eta'])==[0.0,0.0,0.0]),
                     'neighbor_zero_Q64':x['Q_64'] if x and json.loads(x['eta'])==[0.0,0.0,0.0] else '',
                     'neighbor_zero_B63':x['B63_retained'] if x and json.loads(x['eta'])==[0.0,0.0,0.0] else '',
                     'classification':'screening_zero_change_only'})
    write_csv(HERE/'zero_boundary_analysis.csv',zero)

    # Secondary Jdef only applies to promoted parent-canonical transfer; it
    # cannot establish changing neighbor minimum-J canonical representatives.
    jrows=[{'arm_id':r['arm_id'],'anchor_rank':r['anchor_rank'],'neighbor_id':r['neighbor_id'],'offset_steps':r['offset_steps'],'transfer_mean_J_def':r['mean_J_def'],
            'status':'DESCRIPTIVE_ONLY','limitation':'neighbor canonical J_def not queried'} for r in promoted]
    write_csv(HERE/'secondary_jdef_analysis.csv',jrows)

    # Conservative local labels; no one-way result becomes a true switch.
    xbyn={r['neighbor_id']:r for r in cross}
    labels=[]
    for row in pair_q:
        x=xbyn.get(row['neighbor_id']); s8=next(o for o in overlap if o['neighbor_id']==row['neighbor_id'] and o['screening_set']=='S8')
        status='SCREENING_STABLE' if (float(row['mean_abs_delta_Q8'])<=.15 and (math.isnan(float(s8['jaccard'])) or float(s8['jaccard'])>=.5)) else 'SCREENING_DIFFERENT'
        if x and x['B63_retained']: status += '_DIRECTIONAL_B63_RETAINED'
        elif x: status += '_DIRECTIONAL_B63_NOT_RETAINED'
        labels.append({'anchor_rank':row['anchor_rank'],'anchor_state_id':row['anchor_state_id'],'neighbor_id':row['neighbor_id'],'offset_steps':row['offset_steps'],
                       'pair_status':status,'true_basin_switch_supported':False,'reason':'no independently B63-tested neighbor eta/canonical set; bidirectional criterion not met'})
    write_csv(HERE/'local_pair_classifications.csv',labels)
    checks={'screening_expected_unique':expected,'screening_observed_unique':len(raw),'screening_raw_records':len(raw_all),'screening_duplicate_records_excluded':duplicate_screening,'cross_expected_new':expectedx,'cross_observed_new':len(xraw),
            'screening_errors':0,'cross_errors':0,'screening_collision_or_invalid':sum(int(r['outcome'] in ('agent_collision','wall_collision','invalid_action','nan_inf')) for r in raw),
            'cross_collision_or_invalid':sum(int(r['outcome'] in ('agent_collision','wall_collision','invalid_action','nan_inf')) for r in xraw)}
    (HERE/'integrity_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
if __name__=='__main__': main()
