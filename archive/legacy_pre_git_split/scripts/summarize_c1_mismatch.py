"""Matched outcomes of fixed decisions under independent execution noise."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/c1_pretraining_audits/mismatch'
NAMES=['P','P+S','P+g','P+g+S']

def main():
    p=json.loads((OUT/'protocol.json').read_text());summary={};case_rows=[];all_rows=[]
    for panel in p['panels']:
        for seed in p['seeds']:
            rows=[json.loads((OUT/'rows'/f"{panel['rid']}_{seed}_{cid}.json").read_text()) for cid in panel['cids']]
            all_rows.extend(rows);mapping={r['cid']:r for r in rows}
            good=[r for r in rows if r['outcome']['success'] and not r['controller_error']]
            case_rows.append(dict(rid=panel['rid'],seed=seed,panel_coverage=bool(good),
                unknown_coverage=not good and any(r['controller_error'] for r in rows),
                selected={name:mapping[panel['selected'][name]] for name in NAMES},
                successful_alternatives=[r['cid'] for r in good],
                full_predicted_order=sorted(rows,key=lambda r:(panel['prediction_scores'][r['cid']],r['cid']))))
    for name in NAMES:
        rows=[r['selected'][name] for r in case_rows]
        good=[r['outcome']['success'] and not r['controller_error'] for r in rows]
        summary[name]=dict(episodes=len(rows),success=sum(good),success_rate=float(np.mean(good)),
            deadlock=sum(not ok and not r['controller_error'] and r['outcome']['deadlock'] for r,ok in zip(rows,good)),
            timeout=sum(not ok and not r['controller_error'] and not r['outcome']['deadlock'] and not r['outcome']['collision'] for r,ok in zip(rows,good)),
            collision=sum(r['outcome']['collision'] for r in rows),controller_errors=sum(r['controller_error'] is not None for r in rows),
            mean_observed_stagnation_seconds=float(np.mean([r['outcome']['stagnation_seconds'] for r in rows])),
            conditional_ranking_hits=sum(ok and r['panel_coverage'] for ok,r in zip(good,case_rows)),
            panel_coverage_denominator=sum(r['panel_coverage'] for r in case_rows))
    # Cluster bootstrap at initial-state level: four seeds are not256 independent states.
    differences=[]
    for panel in p['panels']:
        cases=[r for r in case_rows if r['rid']==panel['rid']]
        differences.append(np.mean([float(r['selected']['P+g+S']['outcome']['success'])-float(r['selected']['P']['outcome']['success']) for r in cases]))
    rng=np.random.default_rng(20261005);boot=np.mean(rng.choice(differences,(10000,len(differences)),replace=True),axis=1)
    pairwins=pairtotal=0
    for case in case_rows:
        ranked=case['full_predicted_order']
        for i,a in enumerate(ranked):
            if a['controller_error']:continue
            for b in ranked[i+1:]:
                if b['controller_error'] or a['outcome']['success']==b['outcome']['success']:continue
                pairtotal+=1;pairwins+=int(a['outcome']['success'])
    oldwins=oldtotal=0
    for panel in p['panels']:
        rr=[json.loads((ROOT/'results/c1_frozen_unseen_64/traces'/str(panel['rid'])/(cid+'.json')).read_text()) for cid in panel['cids']]
        rr.sort(key=lambda r:(panel['prediction_scores'][r['cid']],r['cid']))
        for i,a in enumerate(rr):
            for b in rr[i+1:]:
                if a['outcome']['success']==b['outcome']['success']:continue
                oldtotal+=1;oldwins+=int(a['outcome']['success'])
    result=dict(branches=len(all_rows),states=len(p['panels']),replicates=len(p['seeds']),summary=summary,
        covered_state_seed_pairs=sum(r['panel_coverage'] for r in case_rows),
        unknown_panel_coverage=sum(r['unknown_coverage'] for r in case_rows),
        full_minus_P_success_difference=float(np.mean(differences)),
        full_minus_P_cluster_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        panel_pairwise_ranking_concordance=pairwins/pairtotal if pairtotal else None,panel_discordant_outcome_pairs=pairtotal,
        predicted_panel_pairwise_concordance=oldwins/oldtotal if oldtotal else None,
        full_rescues_P_failures=sum(not c['selected']['P']['outcome']['success'] and c['selected']['P+g+S']['outcome']['success'] for c in case_rows),
        full_loses_P_successes=sum(c['selected']['P']['outcome']['success'] and not c['selected']['P+g+S']['outcome']['success'] for c in case_rows),
        safety_counts={k:sum(r['safety'][k] for r in all_rows) for k in ['steps','agent_collision_steps','wall_collision_steps','outside_endpoints','cbf_violations','speed_violations']},
        safety_minima={k:min(r['safety'][k] for r in all_rows) for k in ['min_center_separation','min_agent_surface_clearance','min_wall_surface_clearance','min_pair_cbf_residual','min_wall_cbf_residual']},
        all_controller_errors=[r for r in all_rows if r['controller_error']],
        cases=case_rows)
    full=summary['P+g+S'];result['mismatch_screen_pass']=full['success_rate']>=.9 and result['full_minus_P_success_difference']>=-.05
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','all_controller_errors']},indent=2))

if __name__=='__main__':main()
