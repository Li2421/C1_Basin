#!/usr/bin/env python3
"""Read-only exact-cache neighborhood audit plus no-rollout local preflight."""
import csv,json,sqlite3,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
import common
from shared_rollout_db.src.rollout_db import eta_identity

ROOT=common.ROOT;HERE=Path(__file__).resolve().parent
KSW=ROOT/'diagnostics/orthoflow3_mode_free_k_sweep_latency_v1'
DB=ROOT/'shared_rollout_db/rollout.sqlite'
RADII=(.02,.05,.10)

def save_csv(name,rows):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    allrows=list(csv.DictReader((HERE/'frozen_k16_proposal_predictions.csv').open()))
    by=defaultdict(list)
    for r in allrows:by[int(r['episode_index'])].append(r)
    failed=sorted(i for i,rs in by.items() if any(int(r['oracle_coverable']) for r in rs) and
                  any(int(r['original_selected']) and not int(r['B15']) for r in rs))
    assert len(failed)==13
    keys={}
    for r in csv.DictReader((KSW/'candidate_cache_keys.csv').open()):
        if int(r['episode_index']) in failed:keys[int(r['episode_index'])]=(r['state_uid'],r['controller_uid'])
    assert len(keys)==13
    con=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
    seeds=[json.dumps({'future_index':i},separators=(',',':'),sort_keys=True) for i in range(16)]
    centers=[]
    for i in failed:
        rs=sorted(by[i],key=lambda r:int(r['proposal_index']))
        bad=next(r for r in rs if int(r['original_selected']))
        b=np.asarray([float(bad[f'eta{k}']) for k in (1,2,3)])
        good=[r for r in rs if int(r['B15'])]
        nearest=min(good,key=lambda r:np.linalg.norm((np.asarray([float(r[f'eta{k}']) for k in (1,2,3)])-b)/common.DOMAIN_RADIUS))
        oracle=max(good,key=lambda r:(float(r['Q16_lower']),-int(r['proposal_index'])))
        roles=[('bad_false_top1',bad),('nearest_B15',nearest),('oracle_maxQ',oracle)]
        seen=set()
        for role,r in roles:
            j=int(r['proposal_index'])
            if j in seen:continue
            seen.add(j)
            centers.append({'episode_index':i,'center_role':role,'proposal_index':j,'state_uid':keys[i][0],
                            'controller_uid':keys[i][1],
                            'eta':np.asarray([float(r[f'eta{k}']) for k in (1,2,3)]),
                            'true_Q16':float(r['Q16_lower']),'B15':int(r['B15']),
                            'critic_center_mu':float(r['ensemble_mean_p'])})
    # Cache only standard 16 matched seeds under the exact same controller/state key.
    evidence={}
    for i in failed:
        state,ctl=keys[i]
        query='''SELECT e.eta_uid,e.eta1,e.eta2,e.eta3,COUNT(DISTINCT r.seed_key),SUM(r.success)
                 FROM rollout r JOIN eta e ON e.eta_uid=r.eta_uid
                 WHERE r.state_uid=? AND r.controller_uid=? AND r.seed_key IN ('''+','.join('?'*16)+''')
                   AND r.conflict_quarantined=0 AND r.numerical_failure=0 AND r.compatibility_quality='EXACT_REUSE'
                 GROUP BY e.eta_uid'''
        evidence[i]=[{'eta_uid':x[0],'eta':np.asarray(x[1:4],float),'n':int(x[4]),'k':int(x[5])}
                     for x in con.execute(query,(state,ctl,*seeds))]
    h_all=np.load(KSW/'cohort_features.npz')['h_raw']
    virtual_keys=[];virtual_h=[];virtual_eta=[]
    for ci,c in enumerate(centers):
        for radius in RADII:
            vh,ve=common.virtual_neighborhood(np.asarray([h_all[c['episode_index']]]),np.asarray([c['eta']]),radius)
            virtual_keys.append((ci,radius))
            virtual_h.extend(vh);virtual_eta.extend(ve)
    vp=common.predict_members(np.asarray(virtual_h),np.asarray(virtual_eta)).mean(0).reshape(len(virtual_keys),6)
    virtual_lookup={key:pred for key,pred in zip(virtual_keys,vp)}
    center_rows=[];neighbor_rows=[];requests={}
    for ci,c in enumerate(centers):
        i=c['episode_index'];e=c['eta'];ev=evidence[i]
        for radius in RADII:
            found=[];partial=0
            for x in ev:
                d=float(np.linalg.norm((x['eta']-e)/common.DOMAIN_RADIUS))
                if d<=radius+1e-9 and d>1e-9:
                    if x['n']==16:
                        found.append((x,d))
                        neighbor_rows.append({'episode_index':i,'center_role':c['center_role'],'center_proposal_index':c['proposal_index'],
                          'radius':radius,'eta_uid':x['eta_uid'],'distance_normalized':d,'successes16':x['k'],
                          'Q16':x['k']/16,'B15':int(x['k']>=15)})
                    else:partial+=1
            vm=virtual_lookup[(ci,radius)]
            _,ve=common.virtual_neighborhood(np.asarray([h_all[i]]),np.asarray([e]),radius)
            center_rows.append({'episode_index':i,'center_role':c['center_role'],'proposal_index':c['proposal_index'],
              'center_Q16':c['true_Q16'],'center_B15':c['B15'],'center_predicted_Q':c['critic_center_mu'],
              'radius':radius,'exact_Q16_neighbors':len(found),'partial_neighbors':partial,
              'neighbor_true_Q_mean':float(np.mean([x['k']/16 for x,_ in found])) if found else '',
              'neighbor_true_Q_min':float(min(x['k']/16 for x,_ in found)) if found else '',
              'neighbor_B15_fraction':float(np.mean([x['k']>=15 for x,_ in found])) if found else '',
              'critic_virtual_mean':float(vm.mean()),'critic_virtual_min':float(vm.min()),
              'critic_virtual_max':float(vm.max()),'critic_virtual_variance':float(vm.var()),
              'predicted_spike_proxy':int(c['critic_center_mu']-vm.mean()>.15 and c['critic_center_mu']-vm.min()>.3)})
            for v in ve:
                uid=eta_identity(v)[0]
                requests[(c['state_uid'],uid,c['controller_uid'])]={'state_uid':c['state_uid'],'eta_uid':uid,
                    'controller_uid':c['controller_uid'],'seed_keys':seeds}
    save_csv('local_centers.csv',[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in centers])
    save_csv('local_cache_neighbors.csv',neighbor_rows or [{'episode_index':'none'}])
    save_csv('local_neighborhood_audit.csv',center_rows)
    (HERE/'local_probe_preflight_plan.json').write_text(json.dumps({'requests':list(requests.values())},indent=2)+'\n')
    summary={'failure_states':len(failed),'centers':len(centers),'center_roles':{role:sum(c['center_role']==role for c in centers)
             for role in ('bad_false_top1','nearest_B15','oracle_maxQ')},
             'radii':{},'predicted_spike_proxy_centers_at_r0p05':sum(r['predicted_spike_proxy'] for r in center_rows if r['radius']==.05),
             'planned_unique_local_pairs':len(requests),'planned_continuations_before_cache':len(requests)*16,
             'new_rollout':0}
    for radius in RADII:
        rr=[r for r in center_rows if r['radius']==radius]
        summary['radii'][str(radius)]={'centers_with_at_least_one_exact_Q16_neighbor':sum(r['exact_Q16_neighbors']>0 for r in rr),
          'centers_with_at_least_three_exact_Q16_neighbors':sum(r['exact_Q16_neighbors']>=3 for r in rr),
          'bad_centers_with_at_least_three_neighbors':sum(r['center_role']=='bad_false_top1' and r['exact_Q16_neighbors']>=3 for r in rr),
          'good_centers_with_at_least_three_neighbors':sum(r['center_role']!='bad_false_top1' and r['exact_Q16_neighbors']>=3 for r in rr)}
    (HERE/'local_evidence_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
