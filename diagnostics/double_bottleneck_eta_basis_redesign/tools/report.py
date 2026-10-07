"""Complete checks, comparable summaries, figures, and the basis study report."""
import json, os, subprocess, hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from diagnostics.double_bottleneck_eta_basis_redesign.tools.setup import EXPECTED, tree_sha

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'diagnostics/double_bottleneck_eta_basis_redesign'
PREV=ROOT/'diagnostics/double_bottleneck_eta3_seed_robustness'

def read(pattern,directory=None):
    directory=directory or OUT/'raw'
    return [json.loads(s) for p in sorted(directory.glob(pattern)) for s in p.read_text().splitlines() if s.strip()]

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

def stat(xs):
    a=np.asarray([x for x in xs if x is not None],float)
    return {'n':len(a),'median':float(np.median(a)) if len(a) else None,'mean':float(a.mean()) if len(a) else None,
        'p10':float(np.quantile(a,.1)) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None}

def proj(rows):
    if not rows: return {}
    w=np.array([r['episode_steps'] for r in rows]); w=w/w.sum()
    return {'rollouts':len(rows),**{k:float(sum(v*r['correction'][k]['mean'] for v,r in zip(w,rows))) for k in ('raw_norm','executable_norm','projection_removal_norm','removal_ratio')},
        'more_than_half_removed_fraction':float(sum(v*r['correction']['more_than_half_removed_fraction'] for v,r in zip(w,rows)))}

def main():
    full=json.loads((OUT/'full_seed_robustness.json').read_text())
    centers=full['episodes']; center_by_id={e['episode_id']:e for e in centers}
    old=json.loads((PREV/'eta_robust.json').read_text())['episodes']
    count=json.loads((OUT/'followup_counts.json').read_text())
    local=read('pilot_local_shard*.jsonl')+read('full_local_pending_shard*.jsonl')
    expected={(e['episode_id'],k,s) for e in centers if e['Q_max']>=.5 for k in range(16) for s in range(3001,3009)}
    keys=[(r['episode_id'],r['local_index'],r['seed']) for r in local]
    assert len(keys)==len(set(keys)) and set(keys)==expected,'local incomplete/duplicates'
    for r in local: assert r['center_eta_index']==center_by_id[r['episode_id']]['eta_robust']['eta_index']
    lg=defaultdict(list)
    for r in local: lg[r['episode_id'],r['local_index']].append(r)
    local_numerical=[r for r in local if not r.get('scientific_outcome_valid',True)]
    local_stats=[]
    for e in centers:
        if e['Q_max']<.5: continue
        points=[]
        for k in range(16):
            rows=lg[e['episode_id'],k]; valid=[r for r in rows if r.get('scientific_outcome_valid',True)]
            successes=sum(bool(r['success']) for r in valid); invalid=len(rows)-len(valid)
            points.append({'local_index':k,'valid_trials':len(valid),'numerical_solver_failures':invalid,
                'successes':successes,'Q_observed_valid':successes/len(valid) if valid else None,
                'Q_lower_bound':successes/8,'Q_upper_bound':(successes+invalid)/8})
        lower=np.array([p['Q_lower_bound'] for p in points]); upper=np.array([p['Q_upper_bound'] for p in points])
        local_stats.append({'episode_id':e['episode_id'],'eta_index':e['eta_robust']['eta_index'],'local_points':points,
            'fraction_Q_ge_0_50_lower':float(np.mean(lower>=.5)), 'fraction_Q_ge_0_50_upper':float(np.mean(upper>=.5)),
            'fraction_Q_ge_0_75_lower':float(np.mean(lower>=.75)), 'fraction_Q_ge_0_75_upper':float(np.mean(upper>=.75)),
            'median_local_Q_lower':float(np.median(lower)),'median_local_Q_upper':float(np.median(upper))})
    ls={'episodes':local_stats,
        'median_fraction_Q_ge_0_50_lower':stat(r['fraction_Q_ge_0_50_lower'] for r in local_stats),
        'median_fraction_Q_ge_0_50_upper':stat(r['fraction_Q_ge_0_50_upper'] for r in local_stats),
        'median_local_Q_lower':stat(r['median_local_Q_lower'] for r in local_stats),
        'median_local_Q_upper':stat(r['median_local_Q_upper'] for r in local_stats),
        'rollout_records':len(local),'valid_scientific_outcomes':len(local)-len(local_numerical),
        'numerical_solver_failures':len(local_numerical),
        'local_points_on_domain_boundary_fraction':float(np.mean([np.any(np.isclose(r['theta'],[.5,-.5,0],rtol=0,atol=1e-14)|np.isclose(r['theta'],[1.25,.5,.75],rtol=0,atol=1e-14)) for r in local]))}
    save('local_basin_width.json',ls)

    controls=read('P1_pilot_control_shard*.jsonl')+read('full_control_pending_shard*.jsonl')
    assert len(controls)==count['controls_expected']
    assert len({(r['eta_index'],r['episode_id'],r['seed']) for r in controls})==len(controls)
    cg=defaultdict(list)
    for r in controls: cg[r['eta_index']].append(r)
    cp={i:{'eta_index':i,'successes':sum(r['success'] for r in rs),'trials':len(rs),
        'preservation_probability':sum(r['success'] for r in rs)/len(rs),'outcomes':dict(Counter(r['outcome'] for r in rs))} for i,rs in cg.items()}
    oldc=json.loads((PREV/'control_preservation_summary.json').read_text())
    rates=[cp[e['eta_robust']['eta_index']]['preservation_probability'] for e in centers]
    control_summary={'unique_eta':list(cp.values()),'P1_per_episode_center_preservation':stat(rates),
        'P0_per_episode_center_preservation':oldc['median_preservation_rate_over_38_selected_centers'],
        'P1_outcomes':dict(Counter(r['outcome'] for r in controls)),'P1_rollouts':len(controls)}
    save('control_preservation.json',control_summary)
    pilot_seed=json.loads((OUT/'pilot_seed_robustness.json').read_text())['representations']
    pilot_control_comparison={}
    for rep,pattern in [('P0-3D','P0_pilot_control_seed4_cached.jsonl'),('P1-OrthoFlow3','P1_pilot_control_shard*.jsonl')]:
        groups=defaultdict(list)
        for r in read(pattern): groups[r['eta_index']].append(r)
        per_center={i:sum(r['success'] for r in rs)/len(rs) for i,rs in groups.items()}
        pilot_control_comparison[rep]={'selected_center_preservation':stat(per_center[e['eta_robust']['eta_index']] for e in pilot_seed[rep]['episodes'] if e['eta_robust'] is not None),
            'unique_eta_preservation':per_center}
    save('pilot_control_comparison.json',pilot_control_comparison)

    expert=read('expert_span.jsonl',OUT/'offline'); assert len(expert)==96
    valid=[r for r in expert if r['expert_recovery_valid']]
    es={'states':len(expert),'valid_recovery':len(valid),'failure_types':dict(Counter(r.get('expert_error','valid') for r in expert)),
        'zero_residual':stat(r['theta_zero_residual'] for r in valid),
        'fits':{rep:stat(r['fits'][rep]['best_executable_action_residual'] for r in valid) for rep in ('P0-3D','P1-OrthoFlow3')}}
    save('expert_span_summary.json',es)

    audit=read('basis_state_audit.jsonl',OUT/'offline'); geom=json.loads((OUT/'basis_audit.json').read_text())
    eta=np.asarray([p['theta'] for p in json.loads((ROOT/'diagnostics/double_bottleneck_eta3_full_sobol/eta_points.json').read_text())['points']])
    extended={}
    for cls in sorted({r['trajectory_class'] for r in audit}):
        rs=[r for r in audit if r['trajectory_class']==cls]; extended[cls]={}
        for rep,label in (('P0-3D','P0'),('P1-OrthoFlow3','P1')):
            extended[cls][rep]={
                'raw_mean_gram':np.mean([r[label+'_raw_joint']['gram'] for r in rs],axis=0).tolist(),
                'raw_joint_cosines':{f'{i}_{j}':stat(r[label+'_raw_joint']['cosine_gram'][i][j] for r in rs) for i,j in ((0,1),(0,2),(1,2))},
                'executable_goal_flow_cosine_eta0':stat(r['sensitivity_eta_zero'][rep]['executable']['cosine_gram'][0][1] for r in rs),
                'executable_removed_by_dimension_eta0':[stat(r['sensitivity_eta_zero'][rep]['fraction_raw_perturbation_removed_by_dimension'][i] for r in rs) for i in range(3)],
                'unit_basis_norms':[stat(r[label+'_raw_joint']['column_norms'][i] for r in rs) for i in range(3)],
                'raw_rank_counts':dict(Counter(r[label+'_raw_joint']['effective_rank'] for r in rs)),
                'exec_rank_counts_eta0':dict(Counter(r['sensitivity_eta_zero'][rep]['executable']['effective_rank'] for r in rs)),
                'matched_state_common256_raw_correction_norm':stat(float(np.sqrt(max(0,t@np.asarray(r[label+'_raw_joint']['gram'])@t))) for r in rs for t in eta)}
        n=np.array([r['basis_norms']['flow_perp_raw'] for r in rs]); u=np.array([r['basis_norms']['u_safe'] for r in rs])
        extended[cls]['residual_below_one_percent_flow_fraction']=float(np.mean(n<.01*np.maximum(u,1e-12)))
    successful=[r for r in audit if 'P0_sensitivity_successful_eta' in r]
    save('extended_basis_statistics.json',{'classes':extended,'P0_success_eta_exec_cosine':stat(r['P0_sensitivity_successful_eta']['executable']['cosine_gram'][0][1] for r in successful),
        'P0_success_eta_removed_by_dimension':[stat(r['P0_sensitivity_successful_eta']['fraction_raw_perturbation_removed_by_dimension'][i] for r in successful) for i in range(3)]})

    seedrows=read('P1_full_seed16_cached_pilot.jsonl')+read('P1_full_seed_pending_shard*.jsonl')
    p1selected=[r for r in seedrows if r['eta_index']==center_by_id[r['episode_id']]['eta_robust']['eta_index']]
    oldby={e['episode_id']:e for e in old}
    p0selected=[r for r in read('candidate_seed16_shard*.jsonl',PREV/'raw') if r['eta_index']==oldby[r['episode_id']]['eta_robust']['eta_index']]
    ps={'P0_selected_seed_trials':proj(p0selected),'P1_selected_seed_trials':proj(p1selected),
        'P1_successful_trials':proj([r for r in p1selected if r['success']]),'P1_failed_trials':proj([r for r in p1selected if not r['success']])}
    save('projection_statistics.json',ps)

    # Exclude copies from accounting: these patterns are the original new runs.
    scientific=[]
    patterns=['P1_pilot_global_shard*.jsonl','P1_full_global_pending_shard*.jsonl','P1_pilot_seed16_shard*.jsonl','P1_full_seed_pending_shard*.jsonl',
        'pilot_local_shard*.jsonl','full_local_pending_shard*.jsonl','P1_pilot_control_shard*.jsonl','full_control_pending_shard*.jsonl']
    for p in patterns: scientific+=read(p)
    assert len({r['job_id'] for r in scientific})==len(scientific)
    scientific_valid=[r for r in scientific if r.get('scientific_outcome_valid',True)]
    numerical_failures=[r for r in scientific if not r.get('scientific_outcome_valid',True)]
    collisions=sum(bool(r['wall_collision'] or r['agent_collision']) for r in scientific_valid)
    hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,(p,_) in EXPECTED.items()}
    assert all(hashes[n]==h for n,(_,h) in EXPECTED.items())
    hashes['toy_giveway_source_tree']=tree_sha(ROOT/'toy_giveway')
    assert hashes['toy_giveway_source_tree']=='10c8ac15a724ef0c80d8c0e4c0de63d0b28fd89a7956c030734e8a43bd8b4a26'
    env=os.environ.copy();env['C1_PYTHON']='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python';env['JAX_PLATFORMS']='cpu'
    test=subprocess.run(['bash','scripts/test.sh'],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (OUT/'logs/regression.log').write_text(test.stdout); assert test.returncode==0
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for _,(p,h) in EXPECTED.items())
    assert tree_sha(ROOT/'toy_giveway')==hashes['toy_giveway_source_tree']
    save('regression_results.json',{'passed':True,'returncode':test.returncode,'frozen_hashes':hashes,'all_hashes_unchanged':True})
    gs=subprocess.run(['git','status','--short'],cwd=ROOT,text=True,stdout=subprocess.PIPE,check=True)
    save('repository_status.json',{'git_status':gs.stdout,'canonical_integrity_proof':'Frozen SHA-256 checks before and after regression; repository already contains pre-existing tracked edits and untracked scenario directories.'})

    # Dependency-free SVG: the frozen environment intentionally has no plotting package.
    p0=sorted([e['Q_max'] for e in old]+[0]*23); p1=sorted(e['Q_max'] for e in centers)
    def points(values,x0,y0,w,h):
        return ' '.join(f'{x0+w*i/max(1,len(values)-1):.2f},{y0+h*(1-v):.2f}' for i,v in enumerate(values))
    vals=np.array([r['fraction_Q_ge_0_50_lower'] for r in local_stats]); hist,_=np.histogram(vals,bins=np.linspace(0,1,17))
    bars=[]
    for i,n in enumerate(hist):
        height=220*n/max(1,hist.max()); bars.append(f'<rect x="{570+i*24}" y="{270-height:.2f}" width="20" height="{height:.2f}" fill="#4c78a8"/>')
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="350" viewBox="0 0 1000 350">
<rect width="1000" height="350" fill="white"/><g stroke="#222" fill="none"><path d="M60 50V270H500"/><path d="M570 50V270H955"/></g>
<polyline points="{points(p0,60,50,440,220)}" fill="none" stroke="#e45756" stroke-width="2"/>
<polyline points="{points(p1,60,50,440,220)}" fill="none" stroke="#4c78a8" stroke-width="2"/>{''.join(bars)}
<g font-family="sans-serif" font-size="13" fill="#222"><text x="60" y="32">Q_max across 16 MACFlow seeds</text><text x="570" y="32">Conservative local robust-fraction distribution</text>
<text x="80" y="295">episodes sorted by Q_max</text><text x="590" y="295">fraction of local eta with Q >= 0.50</text>
<text x="300" y="72" fill="#e45756">P0</text><text x="340" y="72" fill="#4c78a8">OrthoFlow3</text></g></svg>'''
    (OUT/'figures/robustness_comparison.svg').write_text(svg)

    fullg=json.loads((OUT/'full_global_summary.json').read_text()); pilot=json.loads((OUT/'pilot_seed_robustness.json').read_text())
    shared=Counter(i for e in fullg['P1-OrthoFlow3']['episodes'] for i in e['successful_eta_indices'])
    save('shared_eta_descriptive.json',{'single_seed_global_coverage':dict(shared),
        'selected_center_counts':dict(Counter(e['eta_robust']['eta_index'] for e in centers)),
        'warning':'Global coverage uses original episode sampling seeds; selected-center seed trials do not evaluate every shared eta on every state.'})
    summary={'decision':'USE-ORTHOFLOW3','P0_existence':38,'P1_existence':fullg['P1-OrthoFlow3']['basin_existence'],
        'P0_Q_positive_median':float(np.median([e['Q_max'] for e in old])), 'P1_Q_median':full['median_Q_max'],
        'P1_Q_counts':full['counts'],'local_fraction_median_lower':ls['median_fraction_Q_ge_0_50_lower']['median'],
        'local_fraction_median_upper':ls['median_fraction_Q_ge_0_50_upper']['median'],
        'control_preservation_median':stat(rates)['median'],'new_rollout_records':len(scientific),
        'valid_scientific_outcomes':len(scientific_valid),'numerical_solver_failures':len(numerical_failures),
        'collisions_among_valid_outcomes':collisions,'hashes_unchanged':True,'regression_passed':True}
    save('FINAL_SUMMARY.json',summary)
    pg=ps['P0_selected_seed_trials']; ng=ps['P1_selected_seed_trials']
    b=geom['by_trajectory_class']
    report=f'''# Double-Bottleneck eta basis comparison

## 1. P0 basis redundancy audit

P0 is exactly `eta_g bound(goals-positions) + eta_s u_safe + eta_r mean_j bound(p_i-p_j)`.
The row bound preserves direction and caps norm at 0.5 m/s; relations point away from other agents and are bounded before averaging over three others. At N=2 this is the single-opponent basis. Both projections retain their original constraints and speed limits.

The pilot was sampled uniformly without P0-positive/negative labels: 12 of 61 targets, 12 of 24 controls. Four uniform action-state indices per trajectory yielded 96 baseline anchors. The same four-anchor rule was applied to all three stored eta-success traces (12 states). These stored eta-success traces were selected in the earlier study as representatives, so that stratum is descriptive and not population-representative.

| Trajectory class | States | Median per-agent goal/Flow cosine | Median joint goal/Flow cosine | P0/P1 raw rank median |
|---|---:|---:|---:|---|
'''
    for cls,v in b.items(): report+=f"| {cls} | {v['states']} | {v['P0_per_agent_cosines']['goal_flow']['cosine']['median']:.4f} | {v['P0_joint_goal_flow_cosine']['median']:.4f} | {v['P0_raw_rank']['median']:.0f}/{v['P1_raw_rank']['median']:.0f} |\n"
    report+='''
All pair cosines, per-agent 2D and joint 8D Gram matrices, singular values, rank thresholds, and effective condition numbers are saved per state in `offline/basis_state_audit.jsonl`. Near-zero vectors are excluded from cosine statistics, rather than treated as aligned.

## 2. Post-projection effective-rank audit

Central differences use 0.015625 times each domain width at eta=0 and registered successful P0 eta. Eta=0 is outside the positive-goal search interval; these unconstrained local probes characterize the operator at baseline and do not add rollout search points. The numerical rank threshold is max(0.001, 0.01*sigma_max). Finite differences describe scale-dependent responses of a nonsmooth projection, not a guaranteed analytic Jacobian.

'''
    for cls,v in b.items(): report+=f"- {cls}: median executable rank at zero P0/P1 = {v['P0_executable_rank_eta0']['median']:.0f}/{v['P1_executable_rank_eta0']['median']:.0f}; effective condition number = {v['P0_executable_condition_eta0']['median']:.3f}/{v['P1_executable_condition_eta0']['median']:.3f}.\n"
    report+=f'''
At successful P0 eta references, median executable rank is {geom['P0_successful_eta_sensitivity']['executable_rank']['median']:.0f}. Per-dimension removed perturbation fractions and executable response cosines are in `extended_basis_statistics.json`. P1 does not demonstrate an increase in median rank at eta=0; its benefit must not be described as simply restoring a missing third rank.

At eta=0, median executable goal/Flow cosine on baseline-success states is 0.7117 for P0 and approximately zero for P1; on timeout states it is 0.6592 versus approximately zero. Around the successful P0 eta references the median is -0.2911, showing that active projection can change direction relationships rather than merely rescale parallel columns. Rank-zero responses are logged explicitly; their cosine is undefined in physical terms even though the stored numerical cosine matrix uses zero for null columns.

## 3. P1-OrthoFlow3 definition and energy comparison

For each agent `r = u_safe - B_goal * dot(u_safe,B_goal)/(||B_goal||² + epsilon²)`, epsilon = 1e-6*max_speed. Then `g = eta_g B_goal + eta_perp c*r + eta_r B_rel`.

The globally fixed c = 3.303687238760696 was computed before P1 rollouts from the 96 baseline anchors: safe-Flow row RMS 0.259393 m/s divided by residual RMS 0.0785162 m/s. No per-state normalization is used. The smooth zero-goal fallback approaches u_safe, with no invented direction. An exactly goal-aligned nonzero Flow has only a regularization-sized residual. No sampled scaled residual was <=1e-6 m/s; relative near-degeneracy fractions are also saved.

The same coefficient domain [0.5,1.25] × [-0.5,0.5] × [0,0.75] and same 256 points were used. RMS calibration matches a typical unit basis scale, not a pointwise energy cap. Scaling can amplify individual residuals; the experiment tests orthogonalization plus this fixed calibration together. It cannot attribute all improvement to orthogonality alone.

Selected-center trial step-weighted raw/executed mean norms: P0 {pg['raw_norm']:.4f}/{pg['executable_norm']:.4f} m/s; P1 {ng['raw_norm']:.4f}/{ng['executable_norm']:.4f} m/s. Second-projection removal norms: {pg['projection_removal_norm']:.4f} versus {ng['projection_removal_norm']:.4f}. These use different trajectories and selected centers and are descriptive, not matched-state causal comparisons.

## 4. Pilot P0 vs P1 comparison

Common-256 single-seed basin existence: P0 10/12, P1 12/12. Median positive fractions: 1/256 versus 6/256. Cached P0 rollouts have identical states, seeds, eta points, horizon, and pipeline. Diagnostic P0 arithmetic passed 100 exact canonical-equivalence checks.

## 5. Seed-robustness comparison

Candidate rule: all successful global points when <=8; otherwise deterministic top eight by successful 8-neighbor count, distance to zero, eta index. Both use seeds 2001–2016. Thus the maximum effort per episode is identical; actual candidate counts vary with basin size.

Pilot median Q_max: P0 0.1875, P1 1.0. P0 counts at Q>=0.25/0.50/0.75: 5/0/0; P1: 12/12/12. This triggered the pre-registered full study via median gain >=0.15 and >=3 newly robust cases.

## 6. Local basin-width comparison

Centers with Q_max>=0.50 receive exactly 16 local Sobol points, normalized radius 0.05 per coordinate, clipped to the fixed domain, and seeds 3001–3008. P0 has no eligible centers. P1 has {len(local_stats)} eligible centers. The median fraction of local points guaranteed to have Q>=0.50 is {ls['median_fraction_Q_ge_0_50_lower']['median']:.4f}; its possible upper-bound median is {ls['median_fraction_Q_ge_0_50_upper']['median']:.4f}. Median per-episode local Q lies between {ls['median_local_Q_lower']['median']:.4f} and {ls['median_local_Q_upper']['median']:.4f}. This independent seed set checks selection optimism of Q_max, although these remain measured local neighborhoods, not continuous volume certificates.

Exactly {len(local_numerical)} of 7,808 registered local rollouts is `NUMERICAL_SOLVER_FAILURE`: `fresh_untouched_test|042`, local point 3, seed 3004, at step 601 in the second projection. After the initial rejection, all three authorized additional invocations reused byte-identical state and nominal action, the same problem, constraints, tolerances, controller, eta, and seed; all failed the unchanged certificate. It is excluded from task outcomes. Local statistics above use strict success lower/upper bounds, so it is never imputed as success or task failure. The lower and upper threshold classifications coincide for this local point, hence the population median is unaffected.

Clipping places {ls['local_points_on_domain_boundary_fraction']:.2%} of local points on a domain boundary. Thus the local success fraction describes the prescribed clipped perturbation law, not an unbiased three-dimensional volume estimate.

## 7. Control preservation

Every distinct full-study P1 center was tested on all 24 frozen controls with seeds 4001–4004; identical center tuples were reused. Median preservation probability over selected episode centers: P0 {control_summary['P0_per_episode_center_preservation']:.4f}, P1 {stat(rates)['median']:.4f}. Outcomes: {control_summary['P1_outcomes']}. No eta was optimized for controls. Preservation across new seeds is not a paired estimate of damage versus eta=0 under those same new seeds; controls were successful under their original frozen seeds.

On the matched 12-control pilot subset, median preservation across available selected centers is P0 {pilot_control_comparison['P0-3D']['selected_center_preservation']['median']:.4f}, P1 {pilot_control_comparison['P1-OrthoFlow3']['selected_center_preservation']['median']:.4f}. Episodes with no successful global candidate have no center and are excluded from this center-preservation statistic.

## 8. Full P1 results

| Measure | P0 | P1-OrthoFlow3 |
|---|---:|---:|
| Observed basin /61 | 38 | {fullg['P1-OrthoFlow3']['basin_existence']} |
| Median positive sampled fraction | 1/256 | {fullg['P1-OrthoFlow3']['positive_basin_fraction']['median']*256:.0f}/256 |
| Median Q_max among basin-positive episodes | 0.21875 | {full['median_Q_max']:.5f} |
| Q_max>=0.50 /61 | 0 | {full['counts']['0.5']} |
| Q_max>=0.75 /61 | 0 | {full['counts']['0.75']} |

New rollout records: {len(scientific)}; valid scientific task outcomes: {len(scientific_valid)}; numerical solver failures: {len(numerical_failures)}; collision rollouts among valid outcomes: {collisions}. Each registered tuple has a saved record. Counts exclude copied cache rows.

Two global eta points (indices 37 and 73) each cover 60/61 targets under the original episode sampling seeds. The selected centers use only indices 37, 73, and 217. This is evidence of shared structure, but is not a cross-seed certificate for one universal eta. Coarse global fractions remain small even when local neighborhoods around selected centers are robust.

## 9. Expert executable-action span diagnostic

The same 96 uniform baseline anchors were queried with the unchanged expert; {len(valid)} successful continuations were available. No failed anchor was replaced. Equal 1024 Sobol candidates plus bounded Powell refinement (400 evaluations) fit each representation inside the same domain. These are best-found residuals, not certified global minima.

Median executable-action residual: eta=0 {es['zero_residual']['median']}; P0 {es['fits']['P0-3D']['median']}; P1 {es['fits']['P1-OrthoFlow3']['median']}. Recovery failures and all residuals are retained. This conditional, small recoverable subset and one selected expert mode cannot establish full joint-action span sufficiency. Low-dimensional correction need not reproduce an arbitrary expert action to improve task completion.

Offline expert re-query restores sampled positions and velocity with fresh monitor history and an 850-step continuation budget. It tests local recoverability and action fit, not completion within the original episode's remaining time. All scientific controller rollouts retain the original 850-step horizon and terminal semantics.

## 10. Final decision and scientific answers

**USE-ORTHOFLOW3**.

Q1: Goal and Flow bases overlap strongly per agent; joint rank nevertheless usually remains three.
Q2: Their executable effects are compressed further at nonzero successful eta; the saved response matrices quantify overlap and clipping separately.
Q3: Orthogonalization improves conditioning but does not increase median executable rank at zero.
Q4: P1 materially improves measured seed robustness in the matched pilot and the triggered full study.
Q5: The independent-seed local tests establish neighborhood robustness under strict lower-bound accounting; P0 has no eligible robust center. One numerical failure does not change its local point's Q>=0.50 classification.
Q6: Preservation is measured on all 24 controls and reported above; it is not used for selecting eta.
Q7: P1's success removes the immediate need to choose between agent-specific coefficient expansion and missing-direction redesign. The intervention changes the permissible shared-coefficient field as well as conditioning: per-agent removed parallel Flow components have different magnitudes and signs, so this is not a mere invertible change of the original three shared coordinates.

## 11. Single recommended next step

Freeze this OrthoFlow3 definition and test cross-state reuse of its small set of robust centers on fresh held-out stochastic episodes in a later study, before deciding whether a learned state-to-eta mapping is needed. No learning or representation expansion was started here; canonical P0 is retained unchanged pending review.

## Reproducibility and limitations

Source/data/environment/projection/Toy hashes match the frozen reference and the full regression suite passed. Diagnostic source and all jobs/results remain isolated in this directory. The initial PREREGISTRATION `registered_at` literal incorrectly states midnight September 27; tool execution records establish it was written before the September 26 19:08 basis audit and 19:10 pilot launch. This timestamp clerical error is preserved and disclosed; it must not be treated as a cryptographic external preregistration. No sampling, scale, or decision rule was changed using P1 outcomes. State-audit finite-step removal logging was corrected during analysis to measure around its stated center; rollout/controller code was unaffected. CPU execution was used because CUDA initialization failed; evaluation semantics match the preceding CPU studies.

![Robustness comparison](figures/robustness_comparison.svg)
'''
    (OUT/'REPORT.md').write_text(report)
    manifest=json.loads((OUT/'run_manifest.json').read_text()); manifest.update(state='complete',summary=summary,
        files=[{'path':str(p.relative_to(OUT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((OUT/'raw').glob('*.jsonl'))],
        diagnostic_source_hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'tools').glob('*.py'))},
        completed_full_seed_rollouts=len(seedrows),completed_full_local_rollouts=len(local),completed_full_control_rollouts=len(controls))
    save('run_manifest.json',manifest)
    print(json.dumps(summary))

if __name__=='__main__':main()
