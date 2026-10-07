"""Independent raw-data audit. No imports from previous analysis code."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'diagnostics/true_q_geometry'
EVENTS = ['deadlock', 'success', 'timeout', 'collision']

def read(p):
    return json.loads(Path(p).read_text())

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ci(k, n, alpha=.05):
    return [float(beta.ppf(alpha/2, k, n-k+1)) if k else 0.,
            float(beta.ppf(1-alpha/2, k+1, n-k)) if k<n else 1.]

def counts(rows):
    c = Counter(r['outcome'] for r in rows)
    return {'n': len(rows), 'counts': {e:c[e] for e in EVENTS},
            'probabilities': {e:c[e]/len(rows) for e in EVENTS},
            'exact_95_intervals': {e:ci(c[e],len(rows)) for e in EVENTS}}

def comparison(a,b):
    a=sorted(a,key=lambda r:r['flow_seed']); b=sorted(b,key=lambda r:r['flow_seed'])
    assert [r['flow_seed'] for r in a]==[r['flow_seed'] for r in b]
    x=np.array([r['outcome']=='deadlock' for r in a]); y=np.array([r['outcome']=='deadlock' for r in b])
    up=int(np.sum(~x & y)); down=int(np.sum(x & ~y)); n=len(x)
    # Simultaneous marginal exact intervals imply conservative coverage for the difference,
    # even with paired arms; no degenerate Wald/bootstrap intervals at 0 or 1.
    ca=ci(int(x.sum()),n,.025); cb=ci(int(y.sum()),n,.025)
    return {'low':counts(a),'high':counts(b),'delta_Q_D_high_minus_low':float(y.mean()-x.mean()),
            'conservative_95_difference_interval':[cb[0]-ca[1],cb[1]-ca[0]],
            'discordant_low_only':down,'discordant_high_only':up,
            'paired_exact_p':float(binomtest(up,up+down,.5).pvalue) if up+down else 1.}

def arrays(base,r):
    p=base/r['relative_path']; assert sha(p)==r['sha256'],str(p)
    with np.load(p) as d:return {k:d[k] for k in d.files}

def restore(s,cfg):
    from single_integrator.environment import GiveWayEnv
    env=GiveWayEnv(cfg); env.reset(s['positions'])
    env.velocities=s['last_velocity'].copy(); env.step_count=int(s['step'])
    env.distance_history=[np.full(2,np.nan) for _ in range(env.step_count+1)]
    start=int(s['history_start_step'])
    for j,v in enumerate(s['error_history']): env.distance_history[start+j]=v.copy()
    since=int(s['candidate_since']); env.candidate_since=None if since<0 else since
    env.stuck_timer=0. if since<0 else (env.step_count-since)*cfg.dt
    return env

def pairmetrics(a,b,k=None):
    n=min(len(a['event']),len(b['event'])); n=min(n,k) if k else n
    metric={f'{v}_rms':float(np.sqrt(np.mean((a[v][:n]-b[v][:n])**2)))
            for v in ['g','w','u_exec','positions_after']}
    metric.update(overlap_steps=n,
        active_disagreement=float(np.mean(np.any(a['second_active'][:n]!=b['second_active'][:n],axis=1))),
        clip_a=float(np.mean(np.linalg.norm(a['w']-a['u_exec'],axis=(1,2))>1e-8)),
        clip_b=float(np.mean(np.linalg.norm(b['w']-b['u_exec'],axis=(1,2))>1e-8)))
    metric['alias']=metric['u_exec_rms']<1e-8 and metric['positions_after_rms']<1e-8
    return metric

def corr(x,y):
    return None if np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic)

def static():
    from single_integrator.environment import Config
    from single_integrator.evaluate import load_policy
    from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
    m=read(OLD/'manifest.json'); checks=[]
    # Read no old narrative report at this stage.
    for key in ['frozen_sources_and_specs','study_code','raw_indexes','deliverables_and_figures']:
        for r in m[key]:
            checks.append({'path':r['path'],'match':sha(ROOT/r['path'])==r['sha256']})
    assert all(x['match'] for x in checks)
    checkpoint=Path(m['frozen_checkpoint']['path']); assert sha(checkpoint)==m['frozen_checkpoint']['sha256']
    _,metadata=load_policy(checkpoint); cfg=Config(**metadata['evaluation_environment'])
    catalog=read(OLD/'raw/q_map/state_catalog.json'); catalog={r['state_id']:r for r in catalog}
    groups={}; data={}; raw=[]; maxerr=0.; steps=0
    for stage in ['q_map','direction_validation']:
        base=OLD/'raw'/stage; manifest=read(base/'manifest.json'); group=defaultdict(list)
        for r in manifest['records']:
            d=arrays(base,r); sid=r['state_id']; name=r.get('probe_name',r.get('policy_name'))
            assert str(d['outcome'])==r['outcome']==str(d['event'][-1])
            group[(sid,name)].append(r); data[(stage,sid,name,r['flow_seed'])]=d
            statepath=OLD/'raw/q_map'/catalog[sid]['state_file']
            assert sha(statepath)==catalog[sid]['state_file_sha256']
            with np.load(statepath) as s:env=restore(dict(s),cfg)
            corrector=DiagnosticCorrector(DiagnosticPhi(*r['phi']))
            for j,u in enumerate(d['u_exec']):
                expected=corrector(env.observation(),d['u_safe'][j],cfg.max_speed)
                maxerr=max(maxerr,float(np.max(abs(expected-d['g'][j]))))
                assert np.allclose(env.positions,d['positions_before'][j],atol=1e-12,rtol=0)
                _,_,done,info=env.step(u)
                assert info['termination']==str(d['event'][j]),(r['trace_id'],j)
                assert np.allclose(env.positions,d['positions_after'][j],atol=1e-12,rtol=0)
                assert done==(j==len(d['event'])-1)
            steps+=len(d['event']); raw.append(r)
        groups[stage]=group
    assert maxerr<1e-12
    g=groups['q_map']; dg=groups['direction_validation']; oldq=read(OLD/'full_horizon_q_map.json')
    for s in oldq['states']:
        for p in s['policies']:assert counts(g[(s['state_id'],p['probe_name'])])['counts']==p['counts']
    selected=read(OLD/'raw/directional_selection.json')['states']
    validation=[{'state_id':s['state_id'],**comparison(dg[(s['state_id'],'phi_minus')],dg[(s['state_id'],'phi_plus')])} for s in selected]
    qseeds={r['flow_seed'] for rows in g.values() for r in rows}; dseeds={r['flow_seed'] for rows in dg.values() for r in rows}
    control={'raw_trace_count':len(raw),'replayed_physical_steps':steps,'raw_outcome_replay':'PASS',
        'source_hash_checks':checks,'max_corrector_error':maxerr,'old_directional_validation':validation,
        'old_seed_sets_disjoint':not bool(qseeds & dseeds),
        'all_state_source_pair_ids':[r['pair_id'] for r in catalog.values()],
        'selected_source_pair_ids':[catalog[s['state_id']]['pair_id'] for s in selected],
        'independence_limits':['Three selected states are distinct trajectories, not three independent population studies.',
          'All three were selected for large estimated finite differences; replication is conditional on these choices.',
          'Main-map seed19073 also generated outcome-selected source states; reuse of its suffix is conditioned and not a fresh Monte Carlo draw.',
          'Ten states come from eight pair IDs. Temporal offsets are different trajectories, so time-window trend is confounded by state.',
          'Raw artifacts cannot independently prove the claimed predeclaration timestamp.'],
        'provisional_verdict':'TRUE_Q_GEOMETRY_EXISTS_CONDITIONAL_ON_SELECTED_STATES'}
    write('control_geometry_audit.json',control)
    slices=[]
    for sid in catalog:
        names=['goal_m2','goal_m1','p0','goal_p1','goal_p2']; allq=[counts(g[(sid,p)]) for p in names]
        freshq=[counts([r for r in g[(sid,p)] if r['flow_seed']!=19073]) for p in names]
        q=np.array([x['probabilities']['deadlock'] for x in allq]); jump=float(max(abs(np.diff(q))))
        cat='FLAT_OR_SATURATED' if np.ptp(q)==0 else 'BASIN_BOUNDARY_EVIDENCE' if jump>=.75 else 'INSUFFICIENT_DATA'
        slices.append({'state_id':sid,'phi_goal':[-.25,-.125,0,.125,.25],'all_seed_estimates':allq,
            'fresh_only_estimates':freshq,'classification':cat,'primary_fd':float((q[3]-q[1])/.25),
            'secondary_fd':float((q[4]-q[0])/.5),'adjacent_jump':jump})
    write('nonsmoothness_audit.json',{'slices':slices,'support':'Basin-like transitions plausible to strong descriptively; mathematical nonsmoothness unestablished.',
        'limitations':['Only two coarse scales and 4 draws/cell; steep smooth sigmoid is compatible.',
        'Active-set switching does not prove probability discontinuity; projection onto convex sets can be continuous.',
        'Finite-difference magnitudes estimate secants, not demonstrated local derivatives.']})
    changes=Counter(); details=[]; pairs=[]; feature=[]
    for sid in catalog:
        for (ss,p),rows in g.items():
            if ss!=sid:continue
            vals=[]; survivor=[]
            for r in rows:
                d=data[('q_map',sid,p,r['flow_seed'])]; n=min(100,len(d['event']))
                disp=float(np.linalg.norm(d['positions_after'][n-1]-d['positions_before'][0])); vals.append(disp)
                if len(d['event'])>=100:survivor.append(disp)
                if p!='p0':
                    b=data[('q_map',sid,'p0',r['flow_seed'])]
                    if str(b['outcome'])=='deadlock':
                        e=str(d['outcome']); label='TRUE_ESCAPE' if e=='success' else 'TIMEOUT_SUBSTITUTION' if e=='timeout' else 'DELAYED_DEADLOCK' if e=='deadlock' and len(d['event'])>len(b['event']) else 'NO_EFFECT'
                        changes[label]+=1
            feature.append({'state_id':sid,'policy':p,'displacement_censored_100':float(np.mean(vals)),
                'all_four_survive_100':len(survivor)==4,**counts(rows)})
            if p!='p0' and counts(rows)['probabilities']['deadlock']<counts(g[(sid,'p0')])['probabilities']['deadlock']:
                details.append({'state_id':sid,'policy':p,'baseline':counts(g[(sid,'p0')]),'changed':counts(rows)})
        for a,b in [('goal_m1','goal_p1'),('safe_m1','safe_p1'),('rel_m1','rel_p1'),('goal_m2','goal_p2')]:
            if abs(counts(g[(sid,a)])['probabilities']['deadlock']-counts(g[(sid,b)])['probabilities']['deadlock'])<.5:continue
            for seed in sorted(qseeds):
                aa=data[('q_map',sid,a,seed)]; bb=data[('q_map',sid,b,seed)]
                pairs.append({'state_id':sid,'a':a,'b':b,'seed':seed,'full_overlap':pairmetrics(aa,bb),
                    'prefix20':pairmetrics(aa,bb,20),'prefix100':pairmetrics(aa,bb,100)})
    for s in selected:
        sid=s['state_id']
        for seed in sorted(dseeds):
            aa=data[('direction_validation',sid,'phi_minus',seed)]; bb=data[('direction_validation',sid,'phi_plus',seed)]
            pairs.append({'state_id':sid,'a':'phi_minus','b':'phi_plus','seed':seed,'full_overlap':pairmetrics(aa,bb),
                'prefix20':pairmetrics(aa,bb,20),'prefix100':pairmetrics(aa,bb,100)})
    write('timeout_substitution_audit.json',{'historical_changes_from_deadlock_baseline':dict(changes),
        'every_policy_with_lower_Q_than_p0':details,'old_directional_validation':validation,
        'interpretation':'deadlock-control geometry exists, but task-liveness geometry has not yet been found.'})
    write('projection_audit.json',{'paired_trajectories':len(pairs),'aliases':sum(p['full_overlap']['alias'] for p in pairs),
        'classification':'PARTIALLY_COMPRESSES','interpretation':'Nonzero executed separation survives; many input differences attenuated. No projection-removal causal experiment.',
        'pairs':pairs,'full_overlap_limitation':'Unequal episodes compared only until the first arm terminates; no invented continuation after absorption.'})
    within={sid:corr([r['displacement_censored_100'] for r in feature if r['state_id']==sid],
            [r['probabilities']['deadlock'] for r in feature if r['state_id']==sid]) for sid in catalog}
    surv=[r for r in feature if r['all_four_survive_100']]
    write('feature_claim_audit.json',{'rows':feature,'pooled_spearman':corr([r['displacement_censored_100'] for r in feature],[r['probabilities']['deadlock'] for r in feature]),
        'within_state_spearman':within,'uncensored_group_count':len(surv),'uncensored_group_spearman':corr([r['displacement_censored_100'] for r in surv],[r['probabilities']['deadlock'] for r in surv]),
        'finding':'Association reproducible; not sufficient and not a validated early feature.',
        'censoring':'Old 100-step feature uses min(100, termination), so deadlocks at 20/40/80 steps get shorter motion windows. This introduces outcome-dependent observation length.',
        'selection':'Displacement was the strongest of many examined features; no held-out feature validation.'})
    write('multimodality_audit.json',{'strict_status':'NO_MULTIMODAL_EVIDENCE','final_status':'NOT_SUPPORTED',
        'reason':'No separated successful recovery basins with a worse intervening region were sampled. Several low-Q timeout arms do not establish disconnected basins.',
        'success_basins_exist_in_baseline_success_states':True,'successful_recovery_from_baseline_deadlock_observed':False})
    plan={'locked_at_utc':datetime.now(timezone.utc).isoformat(),'selection_rule':'Repeat all three previously selected FD-validation states; preserve their exact saved opposite phi endpoints.',
        'states':selected,'seeds':list(range(92022001,92022033)),'extension_seeds':list(range(92022033,92022065)),
        'initial_new_rollouts':192,'max_steps':sum((850-catalog[s['state_id']]['start_step'])*64 for s in selected),
        'backend':'Slurm GPU Flow, CPU exact projections/environment; no solver rewrite',
        'paired':True,'success_gate':'Each delta_Q lower 95% bound > 0 and paired p < .05/3; extend only one ambiguous pair to 64/arm if needed.',
        'optional_grid':'SKIPPED: 9x9x8 costs 648 extra episodes; insufficient necessity for this audit.',
        'source_checkpoint':str(checkpoint),'environment':metadata['evaluation_environment']}
    assert not set(plan['seeds']) & (qseeds|dseeds)
    write('replication_plan.json',plan)
    write('blind_static_conclusion.json',{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'old_narrative_report_read':False,'geometry':'Supported on selected states by raw counts and disjoint validation seeds.',
        'basin':'Suggestive, cannot establish discontinuity or distinguish a steep smooth transition.',
        'multimodality':'No affirmative evidence','liveness':'Timeout substitution only in deadlock-recovery comparisons.',
        'projection_pairs':len(pairs),'counts':dict(changes)})
    print(json.dumps({'static_audit':'PASS','replayed_traces':len(raw),'steps':steps,'validation':validation,'plan':plan},ensure_ascii=False),flush=True)

if __name__=='__main__':static()
