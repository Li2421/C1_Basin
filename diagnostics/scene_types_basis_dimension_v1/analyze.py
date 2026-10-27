"""Close the frozen supplementary range control, then analyze saved evidence."""
from core import *
from discovery import NAMES,support,mask
from scenes import TITLES,geometry
import csv,collections,subprocess,datetime

def mean(values):return float(np.mean(values))if values else None
def display_time(value):return '—'if value is None else f'{value:.3f}'
def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-half),float(mid+half)]

def stats(rows):
    success=[r for r in rows if r['success']];unknown=sum(r['numerical_failure']for r in rows)
    return dict(n=len(rows),success=len(success),success_rate=len(success)/len(rows)if rows else None,
        success_mean_time=mean([r['completion_time']for r in success]),all_cohort_truncated_mean=mean([r['truncated_time']for r in rows]),
        collision=sum(r['collision']for r in rows),timeout=sum(r['timeout']for r in rows),numerical_unknown=unknown,
        unknown_exception_types=dict(collections.Counter((r.get('error')or{}).get('type','unspecified')for r in rows if r['numerical_failure'])),
        case_wilson95=wilson(len(success),len(rows)),compute_seconds_total=sum(r.get('wall_seconds',0.)for r in rows),compute_seconds_mean=mean([r.get('wall_seconds',0.)for r in rows]),
        solver_internal_failure_events=sum(r.get('solver_failure_count',0)for r in rows),
        restoration_steps=sum(r.get('restoration_steps',0)for r in rows),sampled_speed_violation_steps=sum(r.get('sampled_speed_violation_steps',0)for r in rows),paper_action_speed_clipped_steps=sum(r.get('speed_clipped_steps',0)for r in rows),constraint_violation_steps=sum(r.get('constraint_violation_steps',0)for r in rows),
        static_collision_without_reported_constraint_violation=sum(bool(r.get('collision_types',{}).get('static'))and 'constraint_violation_steps'in r and r['constraint_violation_steps']==0 for r in rows))

def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='')as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)

def descriptors(frozen):
    from single_integrator.environment import segment_distance,point_segment_distance
    features=[];records=[]
    for e in frozen['entries']:
        t=e['type_id']
        if 'scene'not in e:records.append(dict(type_id=t,title=e['title'],available=False,gap=e.get('gap')));features.append(None);continue
        s=e['scene'];c=s['config'];r=c['agent_radius'];geom=geometry(s);bounds=s['bounds'];area=(bounds[2]-bounds[0])*(bounds[3]-bounds[1]);obstacle=0.
        rects=getattr(geom,'rectangles',np.zeros((0,4)))
        if len(rects):
            xx=sorted({bounds[0],bounds[2],*[v for q in rects for v in(q[0],q[2])]});yy=sorted({bounds[1],bounds[3],*[v for q in rects for v in(q[1],q[3])]})
            obstacle=sum((b-a)*(d-cy)for a,b in zip(xx[:-1],xx[1:])for cy,d in zip(yy[:-1],yy[1:])if any(x0<(a+b)/2<x1 and y0<(cy+d)/2<y1 for x0,y0,x1,y1 in rects))
        openings={3:.9,5:.6,8:.3,9:.3,10:.3,11:.3,12:.5,13:.6,14:s['knob'],15:.4,16:.4}
        narrow=(openings.get(t,float('inf'))+s['widen'])/(2*r)
        doors=4 if t==15 else 1 if t in(8,9,10,11)else 0;intersections=1 if t in(2,5,13,14)else 0
        p=np.asarray(s['states'][0]['positions']);g=np.asarray(s['states'][0]['goals']);dirs=g-p;dirs/=np.maximum(np.linalg.norm(dirs,axis=1),1e-30)[:,None]
        counts=dict(head_on=0,crossing=0,same_direction=0);points=[];shared=0;paths=s['initial_paths'][0]
        for i in range(len(p)):
            for j in range(i):
                best=float('inf');point=None
                for a,b in zip(paths[i][:-1],paths[i][1:]):
                    for u,v in zip(paths[j][:-1],paths[j][1:]):
                        dist=float(segment_distance(np.asarray(a),np.asarray(b),np.asarray(u),np.asarray(v)))
                        if dist<best:best=dist;point=(np.asarray(a)+b+u+v)/4
                if best>2*r:continue
                dot=dirs[i]@dirs[j];counts['head_on'if dot<-.7 else'same_direction'if dot>.7 else'crossing']+=1;points.append(point)
                if t in(8,9,10,11):shared+=1
                elif t==15:
                    portals=np.array([[0.,-.8],[0.,.8],[-.8,0.],[.8,0.]])
                    def visits(path,portal):return any(float(point_segment_distance(portal,np.asarray(a),np.asarray(b)))<=.25 for a,b in zip(path[:-1],path[1:]))
                    shared+=int(any(visits(paths[i],q)and visits(paths[j],q)for q in portals))
        total=sum(counts.values());fractions={k:v/total if total else 0. for k,v in counts.items()};dispersion=float(np.std(np.asarray(points),axis=0).dot(np.std(np.asarray(points),axis=0)))if points else 0.
        robot_fraction=len(p)*np.pi*r*r/max(area-obstacle,1e-30)
        vector=[min(narrow,20.),doors,intersections,obstacle/area,robot_fraction,dispersion,fractions['head_on'],fractions['crossing'],fractions['same_direction'],shared]
        features.append(vector);records.append(dict(type_id=t,title=e['title'],available=True,narrowest_opening_over_diameter=narrow if np.isfinite(narrow)else None,doors=doors,intersections=intersections,obstacle_fraction=obstacle/area,robot_fraction=robot_fraction,path_conflict_midpoint_dispersion=dispersion,conflict_pairs=total,traffic_fractions=fractions,shared_narrow_path_pairs=shared,vector=vector,notes='Geometry-only descriptors at state0, no outcomes. Conflict means initial polyline distance <= one diameter, without predicting arrival times. Dispersion uses segment midpoints, not exact time-dependent collision points.'))
    valid=np.asarray([f for f in features if f is not None]);scale=valid.std(0);scale[scale==0]=1.
    matrix=[[None if a is None or b is None else float(np.linalg.norm((np.asarray(a)-b)/scale))for b in features]for a in features]
    result=dict(records=records,standardization_scale=scale,distance_matrix=matrix,metric='Euclidean distance after feature-wise standardization; open-width infinity capped at20 only in distance vectors. This describes differences and does not prove all scenario mechanisms are independent.')
    dump(HERE/'SCENE_DESCRIPTORS.json',result);return result

def paired(rows,reference):
    def k(r):return(r['type_id'],r['controller'],r['state_index'])
    ref={k(r):r for r in reference};pairs=[(r,ref[k(r)])for r in rows if k(r)in ref and r['success']and ref[k(r)]['success']]
    return dict(paired_success_n=len(pairs),method_mean=mean([a['completion_time']for a,b in pairs]),reference_mean=mean([b['completion_time']for a,b in pairs]),mean_time_difference=mean([a['completion_time']-b['completion_time']for a,b in pairs]))

def family_stats(frozen,labels,rows):
    result=[];curves=[];case_rows={};fixed_rows={}
    for j,f in enumerate(frozen['families']):
        grouped={}
        for label in labels[str(j)]:grouped.setdefault((label['type_id'],label['controller'],label['state_index']),[]).append(label)
        selected=[];fixed=[]
        for key,ll in grouped.items():
            ll.sort(key=lambda x:x['candidate']);pool=[rows[x['uid']]for x in ll if x['uid']in rows]
            if not pool:continue
            first=next((r for r in pool if r['success']),pool[0]);selected.append(first)
            base=next((rows[x['uid']]for x in ll if x['candidate']==0 and x['uid']in rows),None)
            if base:fixed.append(base)
        bytype={}
        for t in range(1,18):
            rr=[r for r in selected if r['type_id']==t];ff=[r for r in fixed if r['type_id']==t]
            bytype[str(t)]=dict(oracle=stats(rr),fixed=stats(ff),protocol_denominator=8,enough=sum(r['success']for r in rr)>=6,missing_cases=8-len(rr))
        entire=stats(selected);fixedstat=stats(fixed)
        failures={k:0 for k in ('collision','timeout','numerical_failure')}
        for label in labels[str(j)]:
            if label['uid']in rows:
                for k in failures:failures[k]+=rows[label['uid']][k]
        candidate_ids={x['uid']for x in labels[str(j)]if x['uid']in rows}
        all_cost=sum(rows[uid].get('wall_seconds',0.)for uid in candidate_ids)
        result.append(dict(index=j,columns=f['columns'],column_names=f['column_names'],dimension=f['dimension'],oracle=entire,fixed=fixedstat,enough_types=sum(x['enough']for x in bytype.values()),by_type=bytype,all_candidate_failures=failures,all_candidate_compute_seconds=all_cost,unique_executed_candidates=len(candidate_ids),oracle_time_convention='Completion time of the first successful candidate in frozen order; all16 were actually run. It is not deployed fixed-parameter runtime. Pool costs include all unique saved candidate work; candidates shared across families can reuse an execution.'))
        case_rows[j]=selected;fixed_rows[j]=fixed
        for k in (1,2,4,8,16):
            cover=sum(any(rows[x['uid']]['success']for x in ll if x['candidate']<k and x['uid']in rows)for ll in grouped.values())
            curves.append(dict(family=j,columns='+'.join(f['columns']),K=k,covered=cover,n=len(grouped)))
    csv_write(HERE/'budget_curves.csv',curves);return result,case_rows,fixed_rows

def final_integrity():
    manifest=read(HERE/'BACKEND_BUILD.json');changed=[]
    for record in manifest['sources']:
        if sha(record['source'])!=record['source_sha256']:changed.append(record['source'])
        if sha(record['target'])!=record['target_sha256']:changed.append(record['target'])
    # All frozen execution inputs are verified; independent work outside these
    # imported inputs is observed through the declared parallel directory stats.
    for plan in HERE.glob('*_planned.json'):
        obj=read(plan)
        for c in obj.get('exact_configs',[])[:1]:
            for p,h in c['controller_payload'].get('execution_sources',{}).items():
                if p=='private_harness':continue
                if sha(p)!=h:
                    archived=HERE/'source_versions'/f'{h}.py'
                    if not(Path(p).resolve().is_relative_to(HERE)and archived.exists()and sha(archived)==h):changed.append(p)
    # Include the live coordinator before its atexit hook, then verify every
    # distinct imported read-only input rather than only the execution entrypoints.
    from guard import save_imports
    save_imports()
    imported={};manifest_paths=list((HERE/'import_manifests').glob('*.json'))
    for p in manifest_paths:
        for filename,h in read(p).items():
            if not Path(filename).resolve().is_relative_to(HERE):imported.setdefault(filename,set()).add(h)
    imported_changes=[]
    for filename,hashes in imported.items():
        if not Path(filename).is_file()or hashes!={sha(filename)}:imported_changes.append(filename)
    for record in read(HERE/'PAPER_PORT_INPUTS.json')['files']:
        if sha(record['source'])!=record['source_sha256']:changed.append(record['source'])
        if sha(record['target'])!=record['target_sha256']:changed.append(record['target'])
    accel=read(HERE/'ACCELERATION_INPUTS.json')
    for record in accel['files']:
        for field in('source','target'):
            if sha(record[field])!=record['sha256']:changed.append(record[field])
    if sha(HERE/'acceleration.py')!=accel['loader_sha256']:changed.append(str(HERE/'acceleration.py'))
    if(HERE/'IMPC_ADAPTER_V2.json').exists():
        revised=read(HERE/'IMPC_ADAPTER_V2.json')
        if sha(revised['source'])!=revised['original_sha256']:changed.append(revised['source'])
        if sha(revised['target'])!=revised['target_sha256']:changed.append(revised['target'])
    for filename,h in read(HERE/'PAPER_PORT_INPUTS.json')['vendor_sources_sha256'].items():
        if sha(filename)!=h:changed.append(filename)
    dump(HERE/'IMPORTED_INPUTS_END.json',dict(readonly_imported_files={p:sorted(h)for p,h in sorted(imported.items())},changed=imported_changes,manifest_count=len(manifest_paths),scope='Every imported file outside this experiment, including dependencies. Private mutable analysis and versioned experiment files remain separately tracked by execution manifests.'))
    changed+=imported_changes
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),git_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=ROOT,text=True),git_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=ROOT)).hexdigest(),changed_frozen_execution_inputs=sorted(set(changed)),python_protection_events=(HERE/'PROTECTION_VIOLATION.jsonl').exists(),parallel_directories_stat={str(p):dict(mtime_ns=p.stat().st_mtime_ns,inode=p.stat().st_ino)for p in(ROOT/'diagnostics').iterdir()if p.is_dir()and p!=HERE},import_manifests=len(manifest_paths),readonly_imported_file_count=len(imported))
    dump(HERE/'INTEGRITY_END.json',result);assert not changed and not result['python_protection_events'];return result

def main():
    if(HERE/'GOAL_RANGE_FROZEN.json').exists():
        from goal_range_followup import main as range_control
        range_control()
    try:
        from route_unknown_audit import main as route_unknown_audit
        route_unknown_audit()
    except Exception as exc:
        with(HERE/'GAPS.md').open('a')as f:f.write('\n路线 unknown 保存状态诊断未完成：'+repr(exc)+'\n')
    try:
        from static_audit import main as audit_mechanisms
        audit_mechanisms()
    except Exception as exc:
        with(HERE/'GAPS.md').open('a')as f:f.write('\n静态机制诊断未完成：'+repr(exc)+'\n')
    if(HERE/'STATIC_MECHANISMS.json').exists():
        mechanisms=read(HERE/'STATIC_MECHANISMS.json')['states'];summary={}
        for name in NAMES:
            if name in('G','P','R'):continue
            retained=[x['retained_ratio']for r in mechanisms for x in r['projection_retention'].get(name,[])if'retained_ratio'in x]
            summary[name]=dict(states=len(mechanisms),zero_column_states=sum(r['column_norms'][name]<1e-10 for r in mechanisms),rank_gain_states=sum(r['rank_with_added_column'][name]>r['B3_rank']for r in mechanisms),projection_retained_ratio_median=mean([float(np.median(retained))])if retained else None,projection_probe_unknowns=sum('numerical_unknown'in x for r in mechanisms for x in r['projection_retention'].get(name,[])))
        dump(HERE/'STATIC_MECHANISM_SUMMARY.json',dict(n_states=len(mechanisms),B3_all16_failed_cases=sum(r['B3_Oracle16_success_count_for_case']==0 for r in mechanisms),B3_rank_counts=dict(collections.Counter(str(r['B3_rank'])for r in mechanisms)),unfinished_B3_rank_counts=dict(collections.Counter(str(r.get('unfinished_only_B3_rank'))for r in mechanisms)),columns=summary,meaning='Selected actual failed-trajectory states, one per type/controller; not a prevalence estimate of certified deadlocks. Rank and per-column projection retention are local numerical diagnostics only. Retention is relative to the original safe action, and other active constraints can couple the projected change.'))
    frozen=read(HERE/'SCENES_FROZEN.json');families=read(HERE/'FAMILIES_FROZEN.json');labels=read(HERE/'HOLDOUT_LABELS.json')
    applicable_types={e['type_id']for e in frozen['entries']if e['status']=='applicable'}
    unavailable_titles=[e['title']for e in frozen['entries']if e['status']!='applicable']
    rows={r['rollout_uid']:r for r in(read(p)for p in(HERE/'raw').glob('*.json'))};desc=descriptors(frozen)
    from delivery_audit import main as audit_saved_evidence
    audit_saved_evidence(rows,frozen,labels)
    fs,case_rows,fixed_rows=family_stats(families,labels,rows);M=max(f['enough_types']for f in fs);best_by_dimension={}
    dump(HERE/'PER_CONTROLLER_RESULTS.json',{str(j):{c['type_code']:{mode:stats([r for r in rr if r['controller']==c['type_code']])for mode,rr in [('oracle',case_rows[j]),('fixed',fixed_rows[j])]}for c in controllers()}for j in case_rows})
    stage_summaries=[];previous=set()
    for stage in ('S1','S2','S3','S4'):
        plan_path=HERE/f"{stage}{frozen.get('stage_suffix','')}_planned.json"
        if not plan_path.exists():continue
        configs=read(plan_path)['exact_configs'];uids={c['rollout_uid']for c in configs};rr=[rows[u]for u in uids if u in rows]
        covered={(r['type_id'],r['controller'],r['state_index'])for r in rr if r['success']}
        stage_summaries.append(dict(stage=stage,plan=str(plan_path),requests=len(configs),unique_planned=len(uids),completed=len(rr),candidate_outcomes=stats(rr),covered_cases=len(covered),new_covered_cases=len(covered-previous),cumulative_covered_cases=len(previous|covered)))
        previous|=covered
    dump(HERE/'SEARCH_STAGE_RESULTS.json',dict(stages=stage_summaries,note='Coverage of the current valid-scope S1-S4 manifests only. Calibration safety and matched-midpoint controls are separately reused by support closure. New coverage counts are relative to earlier listed search stages, not a continuous-family optimum.'))
    from discovery import all_rows
    development=all_rows([]);fit_coverage=[]
    # The empty target set is vacuously covered by zero columns, but that is
    # not a scientific success witness for an entirely uncovered scenario.
    # This reporting-only repair changes no frozen pools or execution choices.
    minimum=read(HERE/'MINIMAL_SUPPORTS.json')
    for item in minimum['per_type'].values():
        if not item['covered_cases']:
            item.update(dimension=None,minimal_supports=[],required_columns=None,box_compatible_single_family_cover_found=False,no_success_witness=True)
    dump(HERE/'MINIMAL_SUPPORTS.json',minimum)
    development_by_spec={}
    for r in development:development_by_spec.setdefault(digest(r['spec']),[]).append(r)
    for j,family in enumerate(families['families']):
        candidates=[]
        for k,sp in enumerate(family['candidate_pool']):
            rr=development_by_spec.get(digest(sp),[])
            unique={(r['type_id'],r['controller'],r['state_index']):r for r in rr}
            candidates.append(dict(index=k,observed_cases=len(unique),observed_success=sum(r['success']for r in unique.values()),total_search_cases=families['search_cases'],fully_evaluated=len(unique)==families['search_cases'],fixed=k==0))
        fit_coverage.append(dict(family=j,columns=family['columns'],candidates=candidates))
    dump(HERE/'FIXED_SELECTION_COVERAGE.json',dict(families=fit_coverage,note='S3/S4 only evaluate uncovered cases. Best observed success count is not a common exhaustive validation of every configuration. Fixed-versus-Oracle gaps can include this selection-data limitation and must not alone be treated as evidence that a learned Generator is necessary. No parameters are refit after seeing heldout results.'))
    # Count actual nonzero supports, not the declared containing family. A
    # five-column family can succeed using a two-column sparse candidate.
    linear_uids={x['uid']for name,ll in labels.items()if name.isdigit()or name in('safety','historical_GC2_fixed','B3_goal_zero_face')for x in ll}
    if(HERE/'GOAL_RANGE_LABELS.json').exists():linear_uids|={x['uid']for ll in read(HERE/'GOAL_RANGE_LABELS.json').values()for x in ll}
    witness_by_case={}
    for uid in linear_uids:
        if uid not in rows or not rows[uid]['success']:continue
        r=rows[uid];k=(r['type_id'],r['controller'],r['state_index']);d=len(support(r['spec']))
        item=witness_by_case.setdefault(k,dict(min_effective_dimension=d,witnesses=[]))
        item['min_effective_dimension']=min(item['min_effective_dimension'],d);item['witnesses'].append(dict(uid=uid,columns=list(support(r['spec'])),dimension=d))
    dump(HERE/'EFFECTIVE_DIMENSION_WITNESSES.json',dict(cases=[dict(type_id=k[0],controller=k[1],state_index=k[2],**v)for k,v in sorted(witness_by_case.items())],five_only_among_tested=[list(k)for k,v in witness_by_case.items()if v['min_effective_dimension']==5],meaning='Minimum nonzero coefficient support among actual successful heldout candidate points, including safety/old-fixed/G0 controls. A declared larger family does not make a sparse success higher-dimensional. These finite witnesses do not prove continuous-space necessity.'))
    linear_success=set(witness_by_case);timing_witnesses=[]
    for name in('right_goal_reference','centre_rotation_reference'):
        for x in labels[name]:
            if x['uid']not in rows:continue
            r=rows[x['uid']];k=(r['type_id'],r['controller'],r['state_index'])
            if r['success']and k not in linear_success:timing_witnesses.append(dict(type_id=k[0],controller=k[1],state_index=k[2],reference=name,uid=r['rollout_uid'],original_constraint_violation_steps=r.get('constraint_violation_steps'),speed_excess_steps=r.get('sampled_speed_violation_steps')))
    dump(HERE/'TIMING_REFERENCE_WITNESSES.json',dict(cases=timing_witnesses,meaning='Same C1 safety pipeline succeeds with a stuck-triggered reference where none of the tested fixed linear-correction candidate points succeeded. References also differ in normalization, magnitude and G=0 policy, so this is not a timing-only ablation. It is not proof that the basis cannot represent a successful untested fixed coefficient. Native-paper safety successes are never used as witnesses for the C1 feasible domain.'))
    for d in range(1,6):
        options=[f for f in fs if f['dimension']==d]
        if options:best_by_dimension[str(d)]=min(options,key=lambda f:(-f['enough_types'],-f['oracle']['success'],f['all_candidate_failures']['collision'],f['oracle']['all_cohort_truncated_mean']or float('inf'),f['index']))
    eligible=[f for f in best_by_dimension.values()if f['enough_types']>=M-1]
    recommendation=min(eligible,key=lambda f:(f['dimension'],-f['oracle']['success'],f['all_candidate_failures']['collision'],f['oracle']['all_cohort_truncated_mean']or float('inf')))if eligible else None
    practical=min((f for f in fs if 1<=f['dimension']<=5),key=lambda f:(-f['enough_types'],-f['oracle']['success'],f['dimension']))
    gaps=[t for t in range(1,18)if not any(f['by_type'][str(t)]['enough']for f in fs)]
    verdict=dict(M=M,recommended_dimension=recommendation['dimension']if recommendation else None,recommended_family_index=recommendation['index']if recommendation else None,recommended_columns=recommendation['columns']if recommendation else None,practical_best_le5=practical['columns'],best_by_dimension={d:dict(family_index=f['index'],columns=f['columns'],enough_types=f['enough_types'],oracle_covered=f['oracle']['success'])for d,f in best_by_dimension.items()},basis_coverage_gap_type_ids=gaps,gap_classification={str(t):('geometry_not_instantiable'if next(e for e in frozen['entries']if e['type_id']==t)['status']=='geometry_gap'else'finite_candidate_pool_insufficient')for t in gaps},rule='Unrounded counts: enough>=6/8; M=max enough types among all frozen families; smallest d<=5 whose best >=M-1. No family found is never an impossibility certificate.',families=fs)
    dump(HERE/'VERDICT.json',verdict)
    b3=next((f['index']for f in fs if set(f['columns'])=={'G','P','R'}),None)
    comparison=[]
    for f in fs:
        a=f['index'];pair=paired(fixed_rows[a],fixed_rows[b3])if b3 is not None else{}
        for t in range(1,18):
            b=f['by_type'][str(t)];comparison.append(dict(method='family_'+str(a),columns='+'.join(f['columns']),type_id=t,title=TITLES[t],dimension=f['dimension'],oracle_success=b['oracle']['success'],oracle_n=b['oracle']['n'],fixed_success=b['fixed']['success'],fixed_n=b['fixed']['n'],oracle_success_mean_time=b['oracle']['success_mean_time'],fixed_success_mean_time=b['fixed']['success_mean_time'],oracle_truncated_mean=b['oracle']['all_cohort_truncated_mean'],fixed_truncated_mean=b['fixed']['all_cohort_truncated_mean'],fixed_collisions=b['fixed']['collision'],fixed_timeouts=b['fixed']['timeout'],fixed_unknown=b['fixed']['numerical_unknown']))
        f['fixed_paired_success_against_B3']=pair
        f['oracle_paired_success_against_B3']=paired(case_rows[a],case_rows[b3])if b3 is not None else{}
    dump(HERE/'PAIRED_SUCCESS_TIMES.json',{str(f['index']):dict(fixed=f['fixed_paired_success_against_B3'],oracle_first_success=f['oracle_paired_success_against_B3'])for f in fs})
    jobids=set()
    for p in HERE.glob('*_jobs.json'):
        for record in read(p):
            if isinstance(record,dict)and'jobid'in record:jobids.add(str(record['jobid']))
    try:
        if jobids:
            accounting=subprocess.check_output(['sacct','-j',','.join(sorted(jobids,key=int)),'--noheader','--parsable2','--format=JobIDRaw,JobName,State,AllocCPUS,ElapsedRaw,TotalCPU,CPUTimeRAW,MaxRSS,Start,End,AllocTRES'],text=True)
            if not accounting.strip():raise RuntimeError('sacct returned no rows; this server reports Slurm accounting storage is disabled. Cancelled/incomplete CPU consumption cannot be reconstructed from sacct.')
            (HERE/'slurm_accounting.psv').write_text('JobIDRaw|JobName|State|AllocCPUS|ElapsedRaw|TotalCPU|CPUTimeRAW|MaxRSS|Start|End|AllocTRES|\n'+accounting)
    except Exception as exc:
        with(HERE/'GAPS.md').open('a')as f:f.write('\nSlurm成本附表不可用：'+repr(exc)+'\n')
    accounting_path=HERE/'slurm_accounting.psv'
    dump(HERE/'COMPUTE_COSTS.json',dict(families=[dict(index=f['index'],columns=f['columns'],all_candidate_compute_seconds=f['all_candidate_compute_seconds'],unique_executed_candidates=f['unique_executed_candidates'])for f in fs],all_unique_rollout_compute_seconds=sum(r.get('wall_seconds',0.)for r in rows.values()),attempts=len(list((HERE/'attempts').glob('*.json'))),complete_unique_records=len(rows),worker_affinity_counts=dict(collections.Counter(str(len(r.get('worker_cpu',[])))for r in rows.values())),svd_seconds=sum(r.get('dictionary',{}).get('svd_seconds',0.)for r in rows.values()),study_wall_seconds=datetime.datetime.now().astimezone().timestamp()-datetime.datetime.fromisoformat(read(HERE/'INTEGRITY_START.json')['time']).timestamp(),stage_elapsed_seconds={p.stem:read(p).get('elapsed')for p in HERE.glob('*_execution.json')},slurm_accounting=str(accounting_path)if accounting_path.exists()else None,cancelled_incomplete_cpu_seconds=None,note='Pool sums overlap across families. Per-record wall_seconds measures runner.work, including episode/trace I/O but excluding fresh-interpreter module-import startup; per-step times are also saved. Stage elapsed includes dispatch/startup/merging, and some development stages overlapped, so do not sum them as study wall time. This server disables Slurm accounting storage; cancelled/incomplete allocation consumption is unavailable rather than zero.'))
    reference={}
    for name in ('safety','right_goal_reference','centre_rotation_reference','historical_GC2_fixed'):
        rr=[rows[x['uid']]for x in labels[name]if x['uid']in rows]
        reference[name]=dict(overall=stats(rr),by_type={str(t):stats([r for r in rr if r['type_id']==t])for t in range(1,18)})
    if 'B3_goal_zero_face'in labels:
        grouped={}
        for label in labels['B3_goal_zero_face']:grouped.setdefault((label['type_id'],label['controller'],label['state_index']),[]).append(label)
        face_rows=[]
        for key,ll in grouped.items():
            ll.sort(key=lambda x:x['candidate']);pool=[rows[x['uid']]for x in ll if x['uid']in rows]
            if pool:face_rows.append(next((r for r in pool if r['success']),pool[0]))
        reference['B3_goal_zero_face']=dict(K=6,overall=stats(face_rows),by_type={str(t):stats([r for r in face_rows if r['type_id']==t])for t in range(1,18)},scope='Separate Goal=0 boundary ablation. Not part of the dimension verdict; not the continuous widened-box optimum.')
    if(HERE/'GOAL_RANGE_LABELS.json').exists():
        from range_analysis import main as analyze_range
        reference.update({'B3_'+name+'_range_control':value for name,value in analyze_range(rows,stats,case_rows).items()})
    papers={}
    for r in rows.values():
        if r['purpose']=='paper_holdout':papers.setdefault(r['controller'],[]).append(r)
    paper_stats={method:dict(overall=stats(rr),by_type={str(t):stats([r for r in rr if r['type_id']==t])for t in range(1,18)})for method,rr in papers.items()}
    baseline_table=[]
    for method,record in {**reference,**paper_stats}.items():
        for t,value in record['by_type'].items():baseline_table.append(dict(method=method,type_id=int(t),title=TITLES[int(t)],**value))
    csv_write(HERE/'baseline_per_type_results.csv',baseline_table)
    from uncertainty import main as uncertainty
    uncertainty(case_rows,fixed_rows,papers)
    dump(HERE/'PAPER_RESULTS.json',paper_stats);dump(HERE/'REFERENCE_RESULTS.json',reference);csv_write(HERE/'per_type_results.csv',comparison)
    local={}
    for f in fs:
        for mode,rr in [('fixed',fixed_rows[f['index']]),('oracle_first_success',case_rows[f['index']])]:
            values=[]
            for r in rr:
                with np.load(r['trace'])as a:
                    remaining=a['route_remaining']if 'route_remaining'in a else None
                    if remaining is None or not len(remaining):continue
                    distance=remaining.sum(axis=1);hit=np.flatnonzero(distance<=distance[0]-.1)
                    if len(hit):values.append(float(hit[0])*.05)
            local[f"family_{f['index']}_{mode}"]=dict(n_with_progress=len(values),n=len(rr),conditional_mean_seconds=mean(values))
    dump(HERE/'LOCAL_PROGRESS_TIMES.json',dict(definition='First cumulative remaining-route reduction >=0.1m; before-action samples at dt0.05. Diagnostic initial-progress proxy, not a deadlock escape certificate. Conditional mean excludes censored episodes.',methods=local,paper_methods='Native full-mechanism ports do not save route_remaining. Leave local metric unavailable rather than silently substituting another distance.'))
    # Constructive success witnesses for previously uncertified calibration states.
    layers=read(HERE/'FIVE_LAYER.json')
    from runner import case as current_case
    active_scenarios={current_case(e['scene'],controllers()[0],0)['scenario_uid']for e in frozen['entries']if e['status']=='applicable'}
    for t in range(1,18):
        if not layers[str(t)].get('geometry_valid'):continue
        witnesses={};restored={}
        for r in rows.values():
            if r['type_id']!=t or r['solo']is not None or not r['success']or r['scenario_uid']not in active_scenarios or r.get('safety_layer')!='distributed':continue
            if r.get('constraint_violation_steps',0)==0 and r.get('sampled_speed_violation_steps',0)==0:witnesses.setdefault(str(r['state_index']),r['rollout_uid'])
            else:restored.setdefault(str(r['state_index']),r['rollout_uid'])
        layers[str(t)]['constructive_rollout_witnesses']=witnesses
        layers[str(t)]['pipeline_success_with_reported_original_constraint_or_speed_violation']=restored
        layers[str(t)]['unwitnessed_search_states']=[i for i in range(4)if str(i)not in witnesses]
    dump(HERE/'FIVE_LAYER_FINAL.json',layers)
    route_diagnostics=read(HERE/'ROUTE_UNKNOWN_DIAGNOSTICS.json')if(HERE/'ROUTE_UNKNOWN_DIAGNOSTICS.json').exists()else{}
    verdict['coverage_gap_diagnostics']={str(t):dict(classification=verdict['gap_classification'][str(t)],unwitnessed_search_states=layers[str(t)].get('unwitnessed_search_states'),heldout_family_candidate_unknowns=sum(rows[u]['numerical_failure']for u in {x['uid']for name,ll in labels.items()if name.isdigit()for x in ll if x['type_id']==t}if u in rows),saved_route_interface_unknown_records=sum(r['type_id']==t for r in route_diagnostics.get('cases',[])),interpretation='Below6/8 in these pools is not evidence of continuous-basis impossibility. Geometry gaps, route/solver unknowns and lack of original-safety joint witnesses limit causal attribution.')for t in gaps}
    verdict['global_search_support_cover']=dict(found=minimum['box_compatible_global_cover_found'],observed_union_cases=minimum['observed_covered_cases'],best_legal_cases=minimum['best_legal_coverage'],minimum_legal_dimension_for_best_coverage=minimum['best_legal_coverage_dimension'])
    dump(HERE/'VERDICT.json',verdict)
    failures=collections.Counter((r['type_id'],r['controller'],r['outcome'])for r in rows.values())
    active_failures=collections.Counter((r['type_id'],r['controller'],r['purpose'],r['outcome'])for r in rows.values()if r['scenario_uid']in active_scenarios)
    dump(HERE/'FAILURE_TYPES.json',dict(all_archived_counts=[dict(type_id=t,controller=c,outcome=o,n=n)for(t,c,o),n in failures.items()],active_primary_scenario_counts=[dict(type_id=t,controller=c,purpose=p,outcome=o,n=n)for(t,c,p,o),n in active_failures.items()],interpretation='All-archive counts include superseded prototypes, calibration ladders and obsolete IMPC adapter identities; they are not formal performance denominators. Formal heldout performance uses frozen family/profile labels. Timeout is not deadlock. Unwitnessed cases remain jointly uncertified. Solver exceptions and internal nonoptimal status counts are separate. Geometry-rejected forest has zero rollouts.'))
    # A complete compact, unique row index accompanies immutable detailed raw records.
    with(HERE/'candidates_unique.jsonl').open('w')as f:
        for uid,r in sorted(rows.items()):f.write(json.dumps({k:r.get(k)for k in('rollout_uid','scene','type_id','controller','state_index','solo','purpose','spec','success','collision','timeout','numerical_failure','outcome','completion_time','truncated_time','wall_seconds','trace','trace_sha256')},ensure_ascii=False)+'\n')
    # A coordinator was paused at a stage boundary during the geometry audit.
    # Preserve its append stream, then provide a valid unique full-record file.
    import shutil
    if(HERE/'candidates.jsonl').exists():shutil.copyfile(HERE/'candidates.jsonl',HERE/'candidate_stage_append_stream.jsonl')
    with(HERE/'candidates.jsonl').open('w')as f:
        for uid,r in sorted(rows.items()):f.write(json.dumps(dict(r,active_primary_scenario=r['scenario_uid']in active_scenarios,raw_record_file=str(HERE/'raw'/f'{uid}.json')),ensure_ascii=False)+'\n')
    recommended=recommendation or practical;fixed=families['families'][recommended['index']]['fixed'];acceleration_certificate=read(HERE/'ACCELERATION_CERTIFICATE.json')
    dimension_text='；'.join(f"{d}维：{f['enough_types']}类"for d,f in best_by_dimension.items())
    pertype='；'.join((f"{TITLES[t]} {recommended['by_type'][str(t)]['oracle']['success']}/8 对 {recommended['by_type'][str(t)]['fixed']['success']}/8"if t in applicable_types else f"{TITLES[t]} N/A（几何或实现缺口）")for t in range(1,18))
    text=[f"# 十七类场景上的修正基维度研究\n\n按预登记规则推荐：{('未找到满足 M−1 条件的五维以内族，以下另列有限池中实用最佳'if recommendation is None else str(recommended['dimension'])+'维')}，列为 {'、'.join(recommended['column_names'])}。逐类留出 Oracle@16 对固定参数成功数：{pertype}。覆盖缺口：{'、'.join(TITLES[t]for t in gaps)}。各维度最好族达到 6/8 的类型数：{dimension_text}。\n\n",
        f"所有结论限于已冻结、已执行的有限候选池。本轮可评估 {len(applicable_types)}/17 类；{'、'.join(unavailable_titles)}为明确缺口，不算控制失败。各可评估类型采用两个冻结名义控制器；每类仅4个独立物理初态，各控制器共享初态与随机流。推荐族固定点实际非零列为 {'、'.join(NAMES[k]for k in support(fixed))or'无（零附加修正）'}；声明族维度不等于每个成功候选都用了全部列。交叉流旧横列原型被排除，未获确认的尺寸或半径提案没有执行。\n\n",
        '## 场景、标定与五层证据\n\n[完整标定](CALIBRATION.json)、[冻结场景与状态](SCENES_FROZEN.json)、[最终五层证据](FIVE_LAYER_FINAL.json)、[几何与安全边距](GEOMETRY_CONSISTENCY.json)。单机能力门均按 ≥95% 判断；未满足难度与 A* 见证条件的类型按最近档保留，仍明确记录未认证。成功 rollout 可补充构造性联合可解见证；超时不能推出死锁或不可解。\n\n',
        '场景类型来源是几何与交通分类的移植，具体尺寸按本轮 prompt 新建，不是论文原地图复现。[17类场景卡、关键尺寸与逐类文献对应表](SCENE_CARDS.md)逐项说明来源与移植边界。圆周、交叉流、同向流、门、盲角与集结对应 [Francis 等场景清单](https://arxiv.org/abs/2306.16740)；森林的动机对应 [BARN](https://www.cs.utexas.edu/~xiao/BARN/BARN.html)，本轮森林几何被拒绝，没有运行 BARN 官方地图。\n\n',
        '| 场景 | 冻结数量/半径 | 路线引导 | 安全基线成功 | A* 见证 |\n|---|---|---|---|---|\n']
    for e in frozen['entries']:
        if 'scene'not in e:text.append(f"|{e['title']}|几何不成立|—|—|—|\n");continue
        s=e['scene'];m=e['calibration'];text.append(f"|{e['title']}|{s['config']['num_agents']} / {s['config']['agent_radius']}m|{s['route_guidance']}|{m.get('safety_success')}/{m.get('safety_n')}|{m.get('witness_success')}/4|\n")
    route_unknown=read(HERE/'ROUTE_UNKNOWN_DIAGNOSTICS.json')if(HERE/'ROUTE_UNKNOWN_DIAGNOSTICS.json').exists()else{}
    route_points=[p for r in route_unknown.get('cases',[])for p in r['points']]
    text += [f"\n路线接口异常共 {route_unknown.get('n',0)} 条保存记录；其中 {sum(p.get('diagnostic_short_connector_path')is not None for p in route_points)}/{len(route_points)} 个异常机器人位置可用一段满足原可见性检查的短连接接回原路线图。这些诊断路径没有供给控制器，也没有改写 unknown。详见 [路线异常诊断](ROUTE_UNKNOWN_DIAGNOSTICS.json)。\n\n",
        '\n森林排斥证据：每个 0.3m 方块向外膨胀 0.25m 后面积为 0.3²+4×0.3×0.25+π×0.25²；12 个的面积 7.03619m²，必须互不重叠且包含在 2.5m 方场地内，其面积仅 6.25m²。该矛盾与控制器无关。\n\n',
        '![冻结场景示意](scene_overview.svg)\n\n## 场景描述与筛选证据\n\n[描述量与距离矩阵](SCENE_DESCRIPTORS.json) 仅使用几何和初始路径。描述量不同不构成机制独立性的数学证明。[最小支持集](MINIMAL_SUPPORTS.json) 使用全部兼容成功见证，严格区分朝目标列存在时系数下界0.5、删除该列时系数为0的两个参数域；不把删除朝目标列的成功偷偷算入正下界家族。若没有一个合法家族覆盖两域全部已知见证，则明确记录有限池缺口，另保留最大合法覆盖比较族。必需列仅相对于本轮有限池。S4测试横向放大+互斥分离族，也是原三列系数朝目标=0的参数边界消融，不能把该类收益全归于新增空间方向。S1、S2、S3、S4 的候选清单、preflight、执行清单均保留；2,000局补搜和未找到成功不构成不可能性证明。\n\n',
        '### 搜索阶段实测\n\n| 阶段 | 计划请求 | 唯一候选完成 | 成功候选 | 覆盖案例 | 新增覆盖案例 | unknown |\n|---|---|---|---|---|---|---|\n']
    for stage in stage_summaries:
        text.append(f"|{stage['stage']}|{stage['requests']}|{stage['completed']}/{stage['unique_planned']}|{stage['candidate_outcomes']['success']}|{stage['covered_cases']}|{stage['new_covered_cases']}|{stage['candidate_outcomes']['numerical_unknown']}|\n")
    attribution=read(HERE/'MATCHED_SINGLE_COLUMN_ATTRIBUTION.json')if(HERE/'MATCHED_SINGLE_COLUMN_ATTRIBUTION.json').exists()else{}
    text+=['\n[搜索阶段原始统计](SEARCH_STAGE_RESULTS.json)按现行有效场景统计，新增覆盖相对之前搜索阶段。单列共同中点控制见 [MATCHED_SINGLE_COLUMN_ATTRIBUTION.json](MATCHED_SINGLE_COLUMN_ATTRIBUTION.json)：原三列历史16点覆盖 '+str(attribution.get('B3_menu16'))+' 个搜索案例，中点单独覆盖 '+str(attribution.get('midpoint'))+' 个，中点本身新增 '+str(attribution.get('midpoint_new_vs_menu16'))+' 个。新增列的补救案例可重叠，不能直接相加。\n\n| 新增列 | 原三列16点及共同中点均失败时新增成功案例 |\n|---|---|\n']
    for col,value in attribution.get('column_new_vs_B3_menu16_and_midpoint',{}).items():text.append(f"|{NAMES[col]}|{value['n']}|\n")
    text+=['\n### 搜索集逐类最小支持与必需列\n\n这里的最小支持覆盖该类型搜索集中所有已观测可成功案例，不是“留出达到6/8”的同一判据。零维表示不加线性修正已有见证；未找到单一家族时不声称这些列具有数学必需性。\n\n| 场景 | 已覆盖搜索案例 | 有限见证最小维度 | 最小支持集（全称） | 所有最小支持共有列 |\n|---|---|---|---|---|\n']
    minimal=read(HERE/'MINIMAL_SUPPORTS.json')
    for t in range(1,18):
        item=minimal['per_type'].get(str(t))
        if item is None:text.append(f'|{TITLES[t]}|N/A|N/A|几何或实现缺口|N/A|\n');continue
        sets=[' + '.join(NAMES[k]for k in cols)or'零修正'for cols in item['minimal_supports']]
        required=item['required_columns']
        required_text='未找到合法单一家族'if required is None else('、'.join(NAMES[k]for k in required)or'无共同列')
        text.append(f"|{TITLES[t]}|{item['covered_cases']}/8|{item['dimension']if item['dimension']is not None else'未找到'}|{'；'.join(sets)or'未找到'}|{required_text}|\n")
    text.append(f"\n搜索集中已成功案例并集为 {minimal['observed_covered_cases']}/{minimal['search_cases']}；单一合法家族的最大已观测覆盖为 {minimal['best_legal_coverage']}，达到该覆盖的最小见证支持为 {minimal['best_legal_coverage_dimension']} 列。覆盖全部并集的合法单一家族{'已找到'if minimal['box_compatible_global_cover_found']else'未找到'}；朝目标正下界域与删除该列的零系数域不互相嵌套，不能把它们强行并成一个合法固定族。该全局覆盖缺口与留出推荐维度是两个不同问题。\n\n")
    text+=['\n## 留出结果与部署差距\n\n| 修正族 | 维度 | 达到6/8的类型 | Oracle覆盖 | 固定参数成功 | Oracle首次成功均时(s) | 固定成功均时(s) | 固定全队列截断均时(s) |\n|---|---|---|---|---|---|---|---|\n']
    for f in fs:text.append(f"|{' + '.join(f['column_names'])}|{f['dimension']}|{f['enough_types']}/17|{f['oracle']['success']}/{f['oracle']['n']}|{f['fixed']['success']}/{f['fixed']['n']}|{display_time(f['oracle']['success_mean_time'])}|{display_time(f['fixed']['success_mean_time'])}|{display_time(f['fixed']['all_cohort_truncated_mean'])}|\n")
    if b3 is not None:
        original=fs[b3];intervals=read(HERE/'CONFIDENCE_INTERVALS.json');j=recommended['index']
        text.append(f"\n推荐族相对原三列有限池：Oracle覆盖差为 {recommended['oracle']['success']-original['oracle']['success']} 个案例，配对物理初态分块成功率差95%区间 `{intervals[f'family_{j}_oracle_minus_B3']['paired_block_bootstrap95']}`；固定参数成功差为 {recommended['fixed']['success']-original['fixed']['success']} 个案例，对应区间 `{intervals[f'family_{j}_fixed_minus_B3']['paired_block_bootstrap95']}`。两种比较分别回答有限候选覆盖与当前固定点部署表现，不能互相替代。不同族的16点池并不全部嵌套，五维／四维的覆盖差也含候选参数采样差异，不是单独新增一列的因果消融。\n\n")
    text+=['\n![独立留出修正族比较](basis_dimension_results.svg)\n\n逐类留出 Oracle@16 成功数；参照规则为单一冻结策略，每格分母8（N/A为未实例化）。\n\n|场景|'+ '|'.join(' + '.join(f['column_names'])for f in fs)+'|只加安全投影|目标右偏参照|群中心旋转参照|\n|---|'+ '|'.join(['---']*(len(fs)+3))+'|\n']
    for t in range(1,18):
        counts=[str(f['by_type'][str(t)]['oracle']['success'])+'/8'if t in applicable_types else'N/A'for f in fs]
        counts += [str(reference[k]['by_type'][str(t)]['success'])+'/8'if t in applicable_types else'N/A'for k in('safety','right_goal_reference','centre_rotation_reference')]
        text.append('|'+TITLES[t]+'|'+'|'.join(counts)+'|\n')
    text += [f'\n推荐族固定参数：`{json.dumps(fixed,ensure_ascii=False)}`。完整逐场景结果见 [per_type_results.csv](per_type_results.csv)，逐名义控制器分解见 [PER_CONTROLLER_RESULTS.json](PER_CONTROLLER_RESULTS.json)，论文端口与参照规则的逐场景成功和时间见 [baseline_per_type_results.csv](baseline_per_type_results.csv)，预算曲线见 [budget_curves.csv](budget_curves.csv)。参数与族在读取留出结果前冻结。[固定参数选取覆盖](FIXED_SELECTION_COVERAGE.json)逐配置列出搜索案例观测量；S3/S4仅补未覆盖案例，所以“最好已观测配置”不等于对所有配置做过统一全队列调参，部署差距不能单独证明需要学习型Generator。维度推荐使用预登记的留出决策规则，是本测试集上的描述性选择，不是另一次独立泛化确认。\n\n',
        '## 论文对应 baseline\n\n五篇论文的六个机制版本使用上一轮冻结参数和同一实际 SI 物理环境，另以预先冻结的26项开发菜单选择参数，保留两个 profile 分别报告。它们保留论文自身 controller、安全 solver、优先级/临时目标机制，区别于两个冻结 nominal 下的修正列消融。论文端口接收真实目标，不附加 C1 的可见图路线引导；有墙地图上的差异因此还包含全局路径信息差异。Oracle@16 是逐案例选择的有限池上限，而论文 profile 是一套固定参数，两者不构成同等部署能力的直接比较；应同时查看修正族固定参数结果。每个论文 profile 有15类×4初态=60局，C1每个族有两个 nominal、共120案例，不能把分母直接混为同一实验。论文原生 benchmark 与原理论假设未复现，原端口适配限制沿用 [上轮报告](../paper_baselines_20261028/report.md)。原始代码 commit 和依赖见 [端口来源](PAPER_PORT_INPUTS.json)。\n\n',
        '各 profile 的精确参数和开发集计数见 [PAPER_PARAMETERS_FROZEN.json](PAPER_PARAMETERS_FROZEN.json)。逐机器人≥95%的能力门只对两个冻结MPC完成，论文端口没有同等全队列单机门；因此它们的超时也可能来自单机绕障或原生规划适用性，不能全部叫多机死锁消解失败。unknown 包括求解、原生规划预算和接口异常；原始兼容字段仍名为 numerical_failure，不能据字段名将全部异常都归因于数值 solver。unknown 比例较高的端口不足以用于原论文方法的能力排名。\n\n',
        '| 论文机制端口 | 成功 | 碰撞/超时/unknown | success条件平均时间(s) | 全队列截断均时(s) | 内部solver失败事件 |\n|---|---|---|---|---|---|\n']
    registry={x['id']:x for x in read(ROOT/'diagnostics/paper_baselines_20261028/PAPER_REGISTRY.json')['allowed']}
    for method,v in paper_stats.items():
        s=v['overall'];entry=registry[method.split('__')[0]];title='['+entry['name']+']('+entry['source']+')'+('（本轮开发集选参）'if'__development_selected'in method else'（上轮冻结参数）');text.append(f"|{title}|{s['success']}/{s['n']}|{s['collision']}/{s['timeout']}/{s['numerical_unknown']}|{display_time(s['success_mean_time'])}|{display_time(s['all_cohort_truncated_mean'])}|{s['solver_internal_failure_events']}|\n")
    text+=['\n![论文端口结局分解](paper_baseline_results.svg)\n\n']
    face_result=reference.get('B3_goal_zero_face',{}).get('overall',{})
    effective=read(HERE/'EFFECTIVE_DIMENSION_WITNESSES.json')
    if(HERE/'GOAL_RANGE_ABLATION_RESULTS.json').exists():
        ranged=read(HERE/'GOAL_RANGE_ABLATION_RESULTS.json');narrow=ranged['results']['narrow32']['overall'];wide=ranged['results']['wide32']['overall']
        text+=[f"\n### 朝目标参数范围的同预算补充对照\n\n两组原三列各32点，前16点完全相同，后16点成对仅改变朝目标系数。原范围覆盖 {narrow['success']}/{narrow['n']}，放宽到0的结构化范围覆盖 {wide['success']}/{wide['n']}；成功条件平均时间分别为 {display_time(narrow['success_mean_time'])}s、{display_time(wide['success_mean_time'])}s。配对物理初态分块的成功率差95%区间为 `{ranged['paired_physical_block_rate_difference_bootstrap95']}`。新点逐对比较中，降低朝目标系数独有成功 {ranged['paired_point_wide_rescues']} 点次，原系数独有成功 {ranged['paired_point_narrow_rescues']} 点次。\n\n此补充协议在主留出已经启动、但未读取留出结局选点时冻结；不修改七族名单、固定参数或主维度判定。它补足旧轮小规模范围消融和本轮仅6点零边界控制的证据缺口，不能冒充连续参数空间最优。[精确池与预算](GOAL_RANGE_FROZEN.json)、[逐点配对及各族相对放宽三列的补救案例](GOAL_RANGE_ABLATION_RESULTS.json)。\n\n"]
    text += [f"\n原三列朝目标系数固定为0的边界消融（另冻结6点）：覆盖 {face_result.get('success')}/{face_result.get('n')}，不参与维度判定。这只比较边界有限池，不是整个放宽参数盒的最优。实际成功候选的最小非零支持见 [EFFECTIVE_DIMENSION_WITNESSES.json](EFFECTIVE_DIMENSION_WITNESSES.json)；只有五个非零系数点成功、而所有已测低维点均未成功的留出案例数为 {len(effective['five_only_among_tested'])}，仍不能据此证明连续参数空间中五维不可替代。\n\n",
        '经典右手目标偏转与群中心旋转启发式是本轮 prompt 指定的附加参照，结果见 [REFERENCE_RESULTS.json](REFERENCE_RESULTS.json)；它们不冒充 Wang、Chen 或 Merry-Go-Round 的完整论文方法，也不参与维度判定。[TIMING_REFERENCE_WITNESSES.json](TIMING_REFERENCE_WITNESSES.json) 列出触发式规则成功而当前固定线性候选池未覆盖的案例。规则同时在归一化、强度与朝目标系数策略上有差别，因此这不是仅改变阶段时序的因果消融。\n\n',
        '## 时间、置信区间与计算成本\n\n[物理初态分块置信区间](CONFIDENCE_INTERVALS.json) 对每个冻结地图内的4个初态分层重采样，把同一初态的两个 nominal 作为同一块；不把8个案例视为独立样本。成功条件均值只包含成功局；[配对成功时间](PAIRED_SUCCESS_TIMES.json)只在两方法均成功的同一 controller×state 上比较；失败局的截断时间使用完整任务时限。Oracle 完成时间取冻结顺序中第一个成功候选的时间，不把最短成功候选时间充作部署速度。Oracle 的计算成本应累加全部 16 候选，不能只算被选成功局。[计算成本](COMPUTE_COSTS.json)和[局部进展代理时间](LOCAL_PROGRESS_TIMES.json)另存，逐候选 wall_seconds 与完整轨迹均保留。Wilson 95% 区间是案例层描述区间，两个 nominal 的成败相关、每类仅四个独立物理初态，不能解释为八个独立样本的严格总体置信区间。\n\n',
        '## 正确性与限制\n\n[真实流水线审计](PIPELINE_AUDIT.md)记录名义控制器、三列真实尺度、原阶段接口、solver与恢复分支；[真实安全域与机制解释](MECHANISM_NOTES.md)说明原逐机器人责任分摊域严格强于单一相对速度约束，协调刚性旋转并不自动可行。静态数值检查见 [STATIC_MECHANISMS.json](STATIC_MECHANISMS.json) 与 [逐列静态摘要](STATIC_MECHANISM_SUMMARY.json)。整局成功可能来自提前预防，不能自动解释为同一死锁态恢复。\n\n[路线引导等价](ROUTE_EQUIVALENCE.json) 为 50 对完整局；[原三列集成回归](B3_EQUIVALENCE.json) 为两组非零历史参数、4 个初态×2个 nominal；[工程加速认证](ACCELERATION_CERTIFICATE.json) 明确列出启用项、200 对逐字节比较、随机键与稀疏数组检查。原安全 solver、重试、恢复、速度上限、物理 step 与成功判据沿用生产实现。T1 使用 prompt 的名称“目标右侧旋转”，但原代码正系数是逆时针 J，负系数才是顺时针；本轮保持该原定义。\n\n',
        f"加速启用同一稀疏矩阵直接构造、NumPy随机键折叠、远墙代价裁剪和局部远墙约束预筛。200对认证局的原版耗时合计 {acceleration_certificate.get('original_seconds')}s，加速版 {acceleration_certificate.get('accelerated_seconds')}s，差值 {acceleration_certificate.get('original_seconds',0)-acceleration_certificate.get('accelerated_seconds',0)}s；这是一批认证局的测量，含认证检查开销，不是所有场景的统一加速比。额外535632个实际输入链随机键比较没有生成留出任务标签。服务器未启用Slurm accounting，因此取消或未完成尝试的完整CPU消耗无法恢复；逐局成本也不含新解释器的模块导入，不能当作总资源账单。\n\n",
        '超时、碰撞、solver unknown 与内部非最优事件分别统计。某状态没有成功见证，只能说有限搜索未解决；尚不能区分搜索不足、时序不足、Basis 不足或联合不可解。原冻结 16 候选本轮不做阶段切换，不能据此排除时序收益。\n\n',
        '## 本轮未包括\n\n学习型规划器、Generator/Critic 训练、人机互动、非完整约束原生动力学、大地图、50 台以上、可选移动障碍与视频均未纳入。森林需用户修改几何规格后另开实验，不能在本轮偷偷缩小间距或方块。原始数据与 SQLite 保留本地，Git 提交小型代码和报告不等于大型轨迹已异地备份。\n']
    (HERE/'REPORT.md').write_text(''.join(text))
    try:
        subprocess.run(['/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python',str(HERE/'plot.py')],check=True)
    except Exception as exc:
        with(HERE/'GAPS.md').open('a')as f:f.write('\n场景图绘制未完成：'+repr(exc)+'\n')
    audit=final_integrity();executions={p.stem:read(p).get('complete')for p in HERE.glob('*_execution.json')}
    valid_count=sum('scene'in e for e in frozen['entries'])
    dump(HERE/'COMPLETE.json',dict(status=('completed_valid_subset_with_gaps'if valid_count<17 else'completed_finite_screening')if all(executions.values())else'partial_evidence_retained',all_planned_batches_complete=all(executions.values()),new_rollout_records=len(rows),attempts=len(list((HERE/'attempts').glob('*.json'))),geometry_types_evaluable=valid_count,requested_types=17,geometry_gap_types=[e['type_id']for e in frozen['entries']if 'scene'not in e],verdict_sha256=sha(HERE/'VERDICT.json'),report_sha256=sha(HERE/'REPORT.md'),execution_status=executions,integrity_passed=True,scientific_scope='Finite Oracle@16 on the explicit valid subset. Geometry/implementation gaps are not controller failures; no continuous optimum or impossibility proof.'))
    (HERE/'STATUS.md').write_text(f'# 当前状态\n\n本轮可评估子集的有限候选实验已交付，详见 REPORT.md 与 COMPLETE.json。已评估{valid_count}/17类；几何和实现缺口、unknown、未完成批次均单列。交叉流几何选项仍待用户决定。\n')
    print('ANALYSIS DELIVERED',verdict['recommended_columns'],flush=True)

if __name__=='__main__':main()
