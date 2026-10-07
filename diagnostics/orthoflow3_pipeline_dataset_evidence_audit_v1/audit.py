#!/usr/bin/env python3
"""Read-only audit of the exact frozen mode-free generator and W1 critic data."""
from __future__ import annotations

import csv
import json
import sqlite3
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

ROOT = Path('/home/zhihan/research/Basin_C1')
OUT = Path(__file__).resolve().parent
GEN = ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
BASE = ROOT/'diagnostics/orthoflow3_continuous_basin_critic_v1'
STRUCT = ROOT/'diagnostics/orthoflow3_structured_continuous_q_data_v1'
DBMODE = ROOT/'diagnostics/orthoflow3_db_shared_mode_transfer_v1'
HARDDB = ROOT/'diagnostics/orthoflow3_db_generator_necessity_v1'
KSWEEP = ROOT/'diagnostics/orthoflow3_mode_free_k_sweep_latency_v1'
DB = ROOT/'shared_rollout_db/rollout.sqlite'
RADIUS = np.array([.625,.5,.375])


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n')


def writecsv(name, rows):
    if not rows: return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def loadcsv(path): return list(csv.DictReader(path.open()))
def coords(r): return np.asarray([r['eta1'],r['eta2'],r['eta3']],float)


def summary(rows, scenario, split, role):
    states={r['state_uid'] for r in rows}
    groups={r.get('source_group') for r in rows if r.get('source_group')}
    by=defaultdict(list)
    for r in rows: by[r['state_uid']].append(r)
    degree=np.asarray([len(x) for x in by.values()],float)
    q=np.asarray([float(r['empirical_q']) for r in rows])
    return {'role':role,'scenario':scenario,'split':split,'states':len(states),'source_groups':len(groups),
        'pairs':len(rows),'unique_eta':len({r['eta_uid'] for r in rows}),
        'eta_per_state_mean':float(degree.mean()) if len(degree) else 0,
        'eta_per_state_median':float(np.median(degree)) if len(degree) else 0,
        'eta_per_state_min':int(degree.min()) if len(degree) else 0,
        'eta_per_state_p90':float(np.percentile(degree,90)) if len(degree) else 0,
        'eta_per_state_max':int(degree.max()) if len(degree) else 0,
        'B15_empirical_or_certified':sum(int(r['n_trials'])>=16 and float(r['empirical_q'])>=15/16 for r in rows),
        'non_B15_n16':sum(int(r['n_trials'])>=16 and float(r['empirical_q'])<15/16 for r in rows),
        'uncertified_lt16':sum(int(r['n_trials'])<16 for r in rows),
        'positive_q_ge_0p9':int(np.sum(q>=.9)),
        'negative_q_le_0p5':int(np.sum(q<=.5)),
        'intermediate_0p1_to_0p9':int(np.sum((q>.1)&(q<.9))),
        'boundary_0p5_to_0p9375':int(np.sum((q>.5)&(q<15/16))),
        'q_exact_zero':int(np.sum(q==0)),'q_exact_one':int(np.sum(q==1))}


def leakage(name, train, val, test):
    def keys(x,f): return {f(r) for r in x}
    out={'dataset':name}
    for a,b,an,bn in ((train,val,'train','val'),(train,test,'train','test'),(val,test,'val','test')):
        for label,fn in [('state_uid',lambda r:r['state_uid']),('source_group',lambda r:r.get('source_group')),
                         ('state_eta',lambda r:(r['state_uid'],r['eta_uid']))]:
            out[f'{an}_{bn}_{label}_overlap']=len(keys(a,fn)&keys(b,fn))
    return out


def geometry(rows,role,scenario,split):
    by=defaultdict(list)
    for r in rows: by[r['state_uid']].append(r)
    out=[]
    for uid, rr in by.items():
        pos=[r for r in rr if int(r['n_trials'])>=16 and float(r['empirical_q'])>=15/16]
        neg=[r for r in rr if float(r['empirical_q'])<=.5]
        inter=[r for r in rr if .5<float(r['empirical_q'])<15/16]
        zz=np.asarray([coords(r)/RADIUS for r in pos]) if pos else np.empty((0,3))
        if len(zz)>1:
            dd=np.linalg.norm(zz[:,None,:]-zz[None,:,:],axis=-1)
            dd[np.eye(len(zz),dtype=bool)]=np.inf
            minsep=float(np.min(dd)); maxspread=float(np.max(dd[np.isfinite(dd)]))
            # A separated pair is not automatically a disconnected component.
            nsep=int(np.sum(np.triu(dd>=.2,1)))
        else: minsep=maxspread=float('nan');nsep=0
        boundary_pair=False
        if pos and neg:
            zp=np.asarray([coords(r)/RADIUS for r in pos]);zn=np.asarray([coords(r)/RADIUS for r in neg])
            boundary_pair=bool(np.min(np.linalg.norm(zp[:,None,:]-zn[None,:,:],axis=-1))<=.10)
        out.append({'role':role,'scenario':scenario,'split':split,'state_uid':uid,'source_group':rr[0].get('source_group',''),
                    'probe_count':len(rr),'B15_positive_count':len(pos),'negative_count':len(neg),
                    'intermediate_count':len(inter),'known_feasible_fraction':len(pos)/len(rr),
                    'robust_min_separation_norm':minsep,'robust_max_spread_norm':maxspread,
                    'robust_pair_separation_ge_0p2':nsep,'has_near_boundary_pos_neg_0p1':int(boundary_pair),
                    'only_one_robust_point':int(len(pos)==1)})
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base=pq.read_table(BASE/'pair_table.parquet').to_pylist()
    gr={s:{sp:[] for sp in ('train','val','test')} for s in ('Toy','DB')}
    genuse={s:{sp:{'positive':[],'paired_negative':[]} for sp in ('train','val')} for s in ('Toy','DB')}
    for s in ('Toy','DB'):
        for sp in ('train','val','test'):
            candidates=[r for r in base if r['scenario']==s and r['state_split']==sp and
                        0<=r['eta1']<=1.25 and -.5<=r['eta2']<=.5 and 0<=r['eta3']<=.75]
            unique={}
            for r in candidates:
                key=(r['state_uid'],r['eta_uid'],r['controller_uid'])
                if key not in unique or r['n_trials']>unique[key]['n_trials']: unique[key]=r
            gr[s][sp]=list(unique.values())
            if sp!='test':
                positives=[r for r in gr[s][sp] if r['empirical_q']>=.9]
                posstates={r['state_uid'] for r in positives}
                negatives=[r for r in gr[s][sp] if r['empirical_q']<=.5 and r['state_uid'] in posstates]
                genuse[s][sp]={'positive':positives,'paired_negative':negatives}
    structured=pq.read_table(STRUCT/'structured_pair_table.parquet').to_pylist()
    sparse=pq.read_table(STRUCT/'sparse_matched_control.parquet').to_pylist()
    st=[{**r,'eta1':r['eta'][0],'eta2':r['eta'][1],'eta3':r['eta'][2],
         'source_group':f"structured_{r['state_uid']}"} for r in structured if r['matrix_partition']=='TRAIN_TRAIN']
    sval=[{**r,'eta1':r['eta'][0],'eta2':r['eta'][1],'eta3':r['eta'][2],
           'source_group':f"structured_{r['state_uid']}"} for r in structured if r['matrix_partition']=='VAL_VAL']
    stest=[{**r,'eta1':r['eta'][0],'eta2':r['eta'][1],'eta3':r['eta'][2],
            'source_group':f"structured_{r['state_uid']}"} for r in structured if r['matrix_partition']=='TESTSTATE_TESTETA']
    sk={(r['state_uid'],r['eta_uid']) for r in st}
    wide=[r for r in sparse if (r['state_uid'],r['eta_uid']) not in sk]
    critic=st+wide
    state_manifest=json.loads((STRUCT/'toy_state_split.json').read_text())
    group_map={r['state_id']:r['source_group'] for r in state_manifest['states']}
    for rr in (st,sval,stest):
        for r in rr: r['source_group']=group_map.get(r['state_id'],r['source_group'])
    summaries=[];geom=[]
    for s in ('Toy','DB'):
        for sp in ('train','val','test'):
            summaries.append(summary(gr[s][sp],s,sp,'generator_eligible'))
            geom+=geometry(gr[s][sp],'generator_eligible',s,sp)
            if sp!='test':
                p=genuse[s][sp]['positive'];n=genuse[s][sp]['paired_negative']
                summaries.append(summary(p,s,sp,'generator_actual_positive'))
                summaries.append(summary(n,s,sp,'generator_actual_negative'))
    for role,rr,sp in [('critic_structured',st,'train'),('critic_wide',wide,'train'),('critic_actual_combined',critic,'train'),
                       ('critic_val',sval,'val'),('critic_test',stest,'test')]:
        summaries.append(summary(rr,'Toy',sp,role))
        geom+=geometry(rr,role,'Toy',sp)
    writecsv('dataset_summary.csv',summaries)
    writecsv('per_state_geometry.csv',geom)
    leak=[leakage('generator_Toy',gr['Toy']['train'],gr['Toy']['val'],gr['Toy']['test']),
          leakage('generator_DB',gr['DB']['train'],gr['DB']['val'],gr['DB']['test']),
          leakage('critic_Toy',critic,sval,stest)]
    # Hard cohort UID/source and normalized h proximity to current training sets.
    con=sqlite3.connect(DB);con.row_factory=sqlite3.Row
    uid_by_ep={int(r['episode_index']):r['state_uid'] for r in loadcsv(GEN/'candidate_cache_keys.csv') if r['kind']=='old_selector_anchor'}
    hard_uids=set(uid_by_ep.values())
    hard_groups={r['source_group'] for r in con.execute('SELECT DISTINCT source_group FROM state WHERE state_uid IN ('+','.join('?'*len(hard_uids))+')',tuple(hard_uids))}
    gtrain=[r for r in gr['Toy']['train'] if r['empirical_q']>=.9]
    overlap={'hard_cohort_states':len(hard_uids),'hard_cohort_source_groups':len(hard_groups),
             'hard_generator_train_state_uid_overlap':len(hard_uids&{r['state_uid'] for r in gtrain}),
             'hard_generator_train_source_group_overlap':len(hard_groups&{r['source_group'] for r in gtrain}),
             'hard_critic_train_state_uid_overlap':len(hard_uids&{r['state_uid'] for r in critic}),
             'hard_critic_train_source_group_overlap':len(hard_groups&{r['source_group'] for r in critic})}
    hard_h=np.load(GEN/'cohort_features.npz')['h_raw']
    train_h=np.asarray([r['h_raw'] for r in {r['state_uid']:r for r in gtrain}.values()],float)
    mu=np.asarray(json.loads((BASE/'dataset_manifest.json').read_text())['state_normalization']['Toy']['mean'])
    sd=np.asarray(json.loads((BASE/'dataset_manifest.json').read_text())['state_normalization']['Toy']['std'])
    dist,_=cKDTree((train_h-mu)/sd).query((hard_h-mu)/sd)
    overlap['hard_to_generator_train_h_l2_norm_median']=float(np.median(dist))
    overlap['hard_to_generator_train_h_l2_norm_min']=float(np.min(dist))
    overlap['hard_to_generator_train_h_rms_below_0p05']=int(np.sum(dist/np.sqrt(hard_h.shape[1])<.05))
    leak.append({'dataset':'hard_cohort_overlap',**overlap})
    writecsv('leakage_audit.csv',leak)
    # DB category audit, distinguishing known full-panel evidence from missing safety measurements.
    mode=loadcsv(DBMODE/'train_mode_counts.csv')
    by_mode=defaultdict(list)
    for r in mode:by_mode[r['source_group']].append(r)
    fixed_by={g:next((r for r in rr if int(r['mode_id'])==0),None) for g,rr in by_mode.items()}
    db_states={r['source_group']:r for r in gr['DB']['train']}
    dbclass=[]
    for group,rr in by_mode.items():
        fixed=fixed_by[group]
        robust=sum(int(r['successes'])>=15 for r in rr)
        safety=con.execute('''SELECT COUNT(*) n,SUM(r.success) k FROM rollout r
            JOIN state s USING(state_uid) JOIN eta e USING(eta_uid)
            WHERE s.source_group=? AND r.controller_uid=? AND e.eta1=0 AND e.eta2=0 AND e.eta3=0
              AND r.conflict_quarantined=0 AND r.numerical_failure=0''',
            (group,json.loads((BASE/'dataset_manifest.json').read_text())['controllers']['DB'])).fetchone()
        sq=(safety['k']/safety['n']) if safety['n'] else None
        dbclass.append({'source_group':group,'in_generator_train':int(group in db_states),
            'fixed_successes':int(fixed['successes']) if fixed else '',
            'fixed_B15':int(int(fixed['successes'])>=15) if fixed else '',
            'safety_n':safety['n'],'safety_Q':sq,
            'safety_B15':int(sq>=15/16) if safety['n']>=16 else '',
            'robust_modes_of_12':robust,'narrow_basin_proxy':int(robust/12<=.1),
            'no_known_basin':int(robust==0),
            'fixed_fail_adaptive_success':int(bool(fixed) and int(fixed['successes'])<15 and robust>0)})
    writecsv('db_train_hardness.csv',dbclass)
    hard=loadcsv(HARDDB/'discrete_baselines.csv')
    hardgroups={r['source_group'] for r in hard}
    dbtrain_groups={r['source_group'] for r in gr['DB']['train']}
    dbtest_groups={r['source_group'] for r in gr['DB']['test']}
    dbhard={'hard_states':len(hard),'hard_source_groups':len(hardgroups),
        'hard_overlap_generator_train_groups':len(hardgroups&dbtrain_groups),
        'hard_overlap_generator_test_groups':len(hardgroups&dbtest_groups),
        'hard_safety_B63':sum(r['safety_B63']=='True' for r in hard),
        'hard_fixed_B63':sum(r['fixed_B63']=='True' for r in hard),
        'hard_selector_B63':sum(r['selector_B63']=='True' for r in hard),
        'hard_fixed_fail_adaptive_success':sum(r['fixed_B63']!='True' and r['oracle_B63']=='True' for r in hard)}
    dump('db_hard_cohort_overlap.json',dbhard)
    # Proposal shift: exact frozen q(eta|h) samples versus actual W1 training eta.
    frozen=json.loads((KSWEEP/'frozen_proposals.json').read_text())['states']
    frozen_first=json.loads((GEN/'frozen_proposals.json').read_text())['states']
    train_eta=np.asarray([coords(r) for r in critic],float)/RADIUS
    proposals=np.asarray([[frozen_first[i]['eta'][f'sample_{k}'] if k<4 else v['eta'][f'sample_{k}']
                            for k in range(16)] for i,v in enumerate(frozen)],float).reshape(-1,3)/RADIUS
    nn=cKDTree(train_eta).query(proposals)[0]
    train_q=np.asarray([float(r['empirical_q']) for r in critic]);
    sample_q=[]
    old=loadcsv(GEN/'per_state_results.csv')
    new=loadcsv(KSWEEP/'per_state_k.csv')
    failstates=[int(r['episode_index']) for r in new if int(r['K'])==16 and int(r['covered_B15']) and not int(r['critic_B15'])]
    # Exact per-proposal Q for 4 existing samples; additional Qs are reconstructed from local raw in the K-sweep audit.
    for i in range(200):
        sample_q += [float(old[i][f'Q16_sample_{j}']) for j in range(4)]
    proposal_q=np.full((200,16),np.nan)
    for i in range(200):
        for j in range(4): proposal_q[i,j]=float(old[i][f'Q16_sample_{j}'])
    counts=defaultdict(lambda:[0,0])
    for path in (KSWEEP/'raw').glob('shard*.jsonl'):
        for line in path.open():
            r=json.loads(line)
            if r.get('kind','').startswith('sample_'):
                key=(int(r['episode_index']),int(r['kind'].split('_')[1]))
                counts[key][0]+=int(bool(r['success']) and bool(r['scientific_outcome_valid']))
                counts[key][1]+=int(bool(r['scientific_outcome_valid']))
    for (i,j),(k,n) in counts.items():
        if n<15:raise RuntimeError(f'Incomplete frozen proposal {i},{j}: {n}')
        proposal_q[i,j]=k/16  # conservative lower Q for the sole censored seed
    if not np.isfinite(proposal_q).all():raise RuntimeError('Proposal-Q matrix incomplete')
    score=np.asarray([v['all_critic_scores'] for v in frozen],float)
    p=1/(1+np.exp(-np.clip(score,-30,30)))
    train_anchor=np.asarray([list(map(float,(r['eta1'],r['eta2'],r['eta3']))) for r in loadcsv(ROOT/'diagnostics/orthoflow3_shared_eta_codebook_v1/codebook_eta.csv')])/RADIUS
    fixed=np.asarray([frozen_first[i]['eta']['fixed_common'] for i in range(200)],float)/RADIUS
    prop_by_state=proposals.reshape(200,16,3)
    nearest_anchor=np.min(cKDTree(train_anchor).query(proposals)[0]) if len(proposals) else float('nan')
    fail_rows=[]
    for i in failstates:
        chosen=int(np.argmax(score[i]))
        best=int(np.argmax(proposal_q[i]))
        fail_rows.append({'episode_index':i,'critic_choice':chosen,'critic_Q16':proposal_q[i,chosen],
            'oracle_choice':best,'oracle_Q16':proposal_q[i,best],
            'critic_predicted_probability':float(p[i,chosen]),
            'oracle_predicted_probability':float(p[i,best]),
            'critic_selected_nearest_train_eta_norm_dist':float(nn.reshape(200,16)[i,chosen]),
            'oracle_nearest_train_eta_norm_dist':float(nn.reshape(200,16)[i,best]),
            'critic_selected_distance_to_fixed_common_norm':float(np.linalg.norm(prop_by_state[i,chosen]-fixed[i])),
            'oracle_distance_to_fixed_common_norm':float(np.linalg.norm(prop_by_state[i,best]-fixed[i]))})
    writecsv('critic_13_failure_states.csv',fail_rows)
    calib=[]
    for lo,hi in ((0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.0000001)):
        mask=(p>=lo)&(p<hi)
        calib.append({'p_low':lo,'p_high':hi,'n':int(mask.sum()),
                      'predicted_Q_mean':float(p[mask].mean()) if mask.any() else None,
                      'observed_Q16_mean':float(proposal_q[mask].mean()) if mask.any() else None,
                      'B15_fraction':float(np.mean(proposal_q[mask]>=15/16)) if mask.any() else None})
    writecsv('critic_deployment_calibration.csv',calib)
    shift={'critic_train_pairs':len(critic),'critic_train_unique_eta':len({r['eta_uid'] for r in critic}),
        'train_eta_coordinate_quantiles':np.percentile(train_eta,[0,25,50,75,100],axis=0).tolist(),
        'proposal_eta_coordinate_quantiles':np.percentile(proposals,[0,25,50,75,100],axis=0).tolist(),
        'proposal_nearest_train_eta_norm_dist_quantiles':np.percentile(nn,[0,25,50,75,90,95,100]).tolist(),
        'proposal_fraction_nearest_train_eta_gt_0p1':float(np.mean(nn>.1)),
        'proposal_fraction_nearest_train_eta_gt_0p2':float(np.mean(nn>.2)),
        'critic_train_Q_quantiles':np.percentile(train_q,[0,25,50,75,100]).tolist(),
        'K4_proposal_Q_quantiles':np.percentile(sample_q,[0,25,50,75,100]).tolist(),
        'K16_proposal_Q_quantiles':np.percentile(proposal_q,[0,25,50,75,100]).tolist(),
        'K16_proposal_B15_fraction':float(np.mean(proposal_q>=15/16)),
        'K16_proposal_boundary_fraction':float(np.mean((proposal_q>.5)&(proposal_q<15/16))),
        'critic_train_B15_fraction':float(np.mean(np.asarray([int(r['n_trials'])>=16 and r['empirical_q']>=15/16 for r in critic]))),
        'critic_train_q_ge_0p9_fraction':float(np.mean(train_q>=.9)),
        'critic_train_boundary_fraction':float(np.mean((train_q>.5)&(train_q<15/16))),
        'proposal_predicted_probability_quantiles':np.percentile(p,[0,25,50,75,100]).tolist(),
        'proposal_nearest_old_anchor_norm_dist_quantiles':np.percentile(cKDTree(train_anchor).query(proposals)[0],[0,25,50,75,100]).tolist(),
        'proposal_distance_to_fixed_common_norm_quantiles':np.percentile(np.linalg.norm(prop_by_state-fixed[:,None,:],axis=-1),[0,25,50,75,100]).tolist(),
        'critic_train_nearest_old_anchor_norm_dist_quantiles':np.percentile(cKDTree(train_anchor).query(train_eta)[0],[0,25,50,75,100]).tolist(),
        'oracle_success_critic_failure_nearest_train_eta_median':float(np.median([r['oracle_nearest_train_eta_norm_dist'] for r in fail_rows])),
        'all_B15_proposal_nearest_train_eta_median':float(np.median(nn[proposal_q.reshape(-1)>=15/16])),
        'oracle_success_critic_failure_Q16_mean':float(np.mean([r['oracle_Q16'] for r in fail_rows])),
        'critic_failure_selected_Q16_mean':float(np.mean([r['critic_Q16'] for r in fail_rows])),
        'K16_oracle_success_critic_failure_states':failstates}
    dump('critic_proposal_shift.json',shift)
    # Required positive source counts and direct consistency checks.
    aggregate={'generator_training_sha256':json.loads((GEN/'generator_training.json').read_text())['source_sha256'],
      'generator_mode_free':True,'current_critic':'Toy-only secondary_combined W1 pair-equal three-seed ensemble',
      'current_critic_DB_training_pairs':0,'DB_hard_cohort':dbhard,
      'DB_train_mode0_B15':sum(x['fixed_B15']==1 for x in dbclass),
      'DB_train_robust_modes_median':float(np.median([x['robust_modes_of_12'] for x in dbclass])),
      'DB_train_safety_measured_n16':sum(x['safety_n']>=16 for x in dbclass),
      'DB_train_fixed_fail_adaptive_success':sum(x['fixed_fail_adaptive_success'] for x in dbclass),
      'leakage':leak,'proposal_shift':shift}
    dump('audit_summary.json',aggregate)
    print(json.dumps({'written':str(OUT),'DB_train_fixed_B15':aggregate['DB_train_mode0_B15'],
                      'critic_train':len(critic),'hard_overlap':dbhard['hard_overlap_generator_train_groups']}))


if __name__=='__main__':main()
