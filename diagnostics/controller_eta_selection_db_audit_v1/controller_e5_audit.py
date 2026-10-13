"""Controller-matched descriptive contrasts and dense-eta coverage census."""
from pathlib import Path
from itertools import combinations
from collections import defaultdict, Counter
import sqlite3, json
import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
DB=HERE.parents[1]/'shared_rollout_db/rollout.sqlite'

def write(name,x):(HERE/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def main():
    panels=json.loads((HERE/'E0_PANEL_RESULTS.json').read_text())
    index={x['panel_uid']:x for x in json.loads((HERE/'PANEL_INDEX.json').read_text())}
    maxpanel={}
    for p in panels:
        cid=p['controller_uid']
        if cid not in maxpanel or p['n_states']*p['n_eta']>maxpanel[cid]['n_states']*maxpanel[cid]['n_eta']:
            maxpanel[cid]=p
    db=sqlite3.connect(f'file:{DB}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    cm={r['controller_uid']:dict(r) for r in db.execute('SELECT * FROM controller_config')}
    pairs=[]
    for a,b in combinations(maxpanel.values(),2):
        if a['scene']!=b['scene']:continue
        ss=sorted(set(a['state_uids'])&set(b['state_uids']))
        ee=sorted(set(a['eta_uids'])&set(b['eta_uids']))
        if len(ss)<8 or len(ee)<2:continue
        qa=np.load(index[a['panel_uid']]['file'],allow_pickle=False)['Y'].mean(2)
        qb=np.load(index[b['panel_uid']]['file'],allow_pickle=False)['Y'].mean(2)
        aa=qa[np.ix_([a['state_uids'].index(s) for s in ss],[a['eta_uids'].index(e) for e in ee])]
        bb=qb[np.ix_([b['state_uids'].index(s) for s in ss],[b['eta_uids'].index(e) for e in ee])]
        ia=int(np.argmax(aa.mean(0)));ib=int(np.argmax(bb.mean(0)))
        means=[aa.mean(0),bb.mean(0)]
        rho=float(spearmanr(*means).statistic) if np.std(means[0]) and np.std(means[1]) else None
        rng=np.random.default_rng(20261011)
        ix=rng.integers(0,len(ss),(2000,len(ss)))
        da=aa[:,ia]-aa[:,ib];dbb=bb[:,ib]-bb[:,ia]
        pairs.append({'scene':a['scene'],'controller_a':a['controller_uid'],
                      'controller_b':b['controller_uid'],'n_matched_states':len(ss),
                      'n_matched_eta':len(ee),'matched_state_uids':ss,'matched_eta_uids':ee,
                      'same_horizon':cm[a['controller_uid']]['horizon']==cm[b['controller_uid']]['horizon'],
                      'same_flow_checkpoint':cm[a['controller_uid']]['flow_checkpoint_sha256']==cm[b['controller_uid']]['flow_checkpoint_sha256'],
                      'same_safety_config':cm[a['controller_uid']]['safety_config_hash']==cm[b['controller_uid']]['safety_config_hash'],
                      'global_eta_spearman':rho,'best_eta_a':ee[ia],'best_eta_b':ee[ib],
                      'best_eta_differs':bool(ia!=ib),
                      'success_a_own_best':float(aa[:,ia].mean()),
                      'success_a_uses_b_best':float(aa[:,ib].mean()),
                      'success_b_own_best':float(bb[:,ib].mean()),
                      'success_b_uses_a_best':float(bb[:,ia].mean()),
                      'a_own_minus_b_choice_ci95':np.quantile(da[ix].mean(1),[.025,.975]).tolist(),
                      'b_own_minus_a_choice_ci95':np.quantile(dbb[ix].mean(1),[.025,.975]).tolist(),
                      'state_argmax_eta_disagreement_fraction':float((aa.argmax(1)!=bb.argmax(1)).mean())})
    write('CONTROLLER_COMPARISONS.json',pairs)
    by=defaultdict(int);scene=defaultdict(lambda:defaultdict(int))
    for r in db.execute('''SELECT r.controller_uid,r.state_uid,sc.name scene,COUNT(*) k FROM
             (SELECT controller_uid,state_uid,eta_uid FROM rollout
              WHERE conflict_quarantined=0 AND numerical_failure=0
              GROUP BY controller_uid,state_uid,eta_uid HAVING COUNT(*)>=16) r
             JOIN state st ON st.state_uid=r.state_uid
             JOIN scenario sc ON sc.scenario_uid=st.scenario_uid
             GROUP BY r.controller_uid,r.state_uid'''):
        k=r['k'];by[k]+=1
        scene[r['scene']][k]+=1
    hist={str(k):v for k,v in sorted(by.items())}
    dense={'Q16_state_controller_groups':sum(by.values()),'Q16_eta_count_histogram':hist,
           'max_Q16_eta_per_state_controller':max(by),
           'groups_at_least_64_eta_Q16':sum(v for k,v in by.items() if k>=64),
           'groups_at_least_32_eta_Q16':sum(v for k,v in by.items() if k>=32),
           'max_shared_Q16_eta_across_at_least_12_states':max(p['n_eta'] for p in panels),
           'Q16_groups_at_least_64_by_scene':{s:sum(v for k,v in d.items() if k>=64) for s,d in scene.items()},
           'note':'Count criterion is necessary, not sufficient: no grid spacing, boundary coverage, or connected-basin topology was certified.'}
    q64=list(db.execute('''SELECT r.controller_uid,r.state_uid,sc.name scene,st.source_group,COUNT(*) k FROM
             (SELECT controller_uid,state_uid,eta_uid FROM rollout
              WHERE conflict_quarantined=0 AND numerical_failure=0
              GROUP BY controller_uid,state_uid,eta_uid HAVING COUNT(*)>=64) r
             JOIN state st ON st.state_uid=r.state_uid
             JOIN scenario sc ON sc.scenario_uid=st.scenario_uid
             GROUP BY r.controller_uid,r.state_uid'''))
    dense['Q64_state_controller_groups']=len(q64)
    dense['Q64_cells']=sum(r['k'] for r in q64)
    dense['max_Q64_eta_per_state_controller']=max((r['k'] for r in q64),default=0)
    dense['Q64_groups_at_least_64_eta']=[dict(r) for r in q64 if r['k']>=64]
    q64_eta=defaultdict(lambda:defaultdict(set))
    for r in db.execute('''SELECT controller_uid,state_uid,eta_uid FROM rollout
             WHERE conflict_quarantined=0 AND numerical_failure=0
             GROUP BY controller_uid,state_uid,eta_uid HAVING COUNT(*)>=64'''):
        q64_eta[r['controller_uid']][r['state_uid']].add(r['eta_uid'])
    dense['Q64_dense_group_shared_eta']={cid:{'n_states_with_at_least_64_eta':len(groups),
             'full_intersection_eta_count':len(set.intersection(*groups.values()))}
             for cid,allstates in q64_eta.items()
             if (groups:={s:es for s,es in allstates.items() if len(es)>=64})}
    write('E5_DENSE_ETA_AUDIT.json',dense)
    print({'controller_pairs':len(pairs),'state_controller_groups':dense['Q16_state_controller_groups'],
           'max_Q16_eta':dense['max_Q16_eta_per_state_controller'],
           'groups_ge64':dense['groups_at_least_64_eta_Q16']},flush=True)
if __name__=='__main__':main()
