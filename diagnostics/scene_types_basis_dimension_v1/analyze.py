"""Read-only analysis of frozen plans and saved evidence, no rollouts."""
from core import *
from discovery import NAMES,support,mask
from scenes import TITLES,geometry
import csv,collections,subprocess,datetime

def mean(values):return float(np.mean(values))if values else None
def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-half),float(mid+half)]

def stats(rows):
    success=[r for r in rows if r['success']];unknown=sum(r['numerical_failure']for r in rows)
    return dict(n=len(rows),success=len(success),success_rate=len(success)/len(rows)if rows else None,
        success_mean_time=mean([r['completion_time']for r in success]),all_cohort_truncated_mean=mean([r['truncated_time']for r in rows]),
        collision=sum(r['collision']for r in rows),timeout=sum(r['timeout']for r in rows),numerical_unknown=unknown,
        case_wilson95=wilson(len(success),len(rows)),compute_seconds_total=sum(r.get('wall_seconds',0.)for r in rows),compute_seconds_mean=mean([r.get('wall_seconds',0.)for r in rows]),
        solver_internal_failure_events=sum(r.get('solver_failure_count',0)for r in rows))

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
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),git_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=ROOT,text=True),git_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=ROOT)).hexdigest(),changed_frozen_execution_inputs=sorted(set(changed)),python_protection_events=(HERE/'PROTECTION_VIOLATION.jsonl').exists(),parallel_directories_stat={str(p):dict(mtime_ns=p.stat().st_mtime_ns,inode=p.stat().st_ino)for p in(ROOT/'diagnostics').iterdir()if p.is_dir()and p!=HERE},import_manifests=len(list((HERE/'import_manifests').glob('*.json'))))
    dump(HERE/'INTEGRITY_END.json',result);assert not changed and not result['python_protection_events'];return result

def main():
    frozen=read(HERE/'SCENES_FROZEN.json');families=read(HERE/'FAMILIES_FROZEN.json');labels=read(HERE/'HOLDOUT_LABELS.json')
    rows={r['rollout_uid']:r for r in(read(p)for p in(HERE/'raw').glob('*.json'))};desc=descriptors(frozen)
    fs,case_rows,fixed_rows=family_stats(families,labels,rows);M=max(f['enough_types']for f in fs);best_by_dimension={}
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
    dump(HERE/'PAIRED_SUCCESS_TIMES.json',{str(f['index']):f['fixed_paired_success_against_B3']for f in fs})
    dump(HERE/'COMPUTE_COSTS.json',dict(families=[dict(index=f['index'],columns=f['columns'],all_candidate_compute_seconds=f['all_candidate_compute_seconds'],unique_executed_candidates=f['unique_executed_candidates'])for f in fs],all_unique_rollout_compute_seconds=sum(r.get('wall_seconds',0.)for r in rows.values()),attempts=len(list((HERE/'attempts').glob('*.json'))),complete_unique_records=len(rows),svd_seconds=sum(r.get('dictionary',{}).get('svd_seconds',0.)for r in rows.values()),note='Pool sums may overlap across families. Unique sum is completed work across the study; cancelled/incomplete work appears separately in Slurm accounting.'))
    reference={name:stats([rows[x['uid']]for x in labels[name]if x['uid']in rows])for name in ('safety','right_goal_reference','centre_rotation_reference')}
    papers={}
    for r in rows.values():
        if r['purpose']=='paper_holdout':papers.setdefault(r['controller'],[]).append(r)
    paper_stats={method:dict(overall=stats(rr),by_type={str(t):stats([r for r in rr if r['type_id']==t])for t in range(1,18)})for method,rr in papers.items()}
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
    for t in range(1,18):
        if not layers[str(t)].get('geometry_valid'):continue
        witnesses={}
        for r in rows.values():
            if r['type_id']==t and r['solo']is None and r['success']:witnesses.setdefault(str(r['state_index']),r['rollout_uid'])
        layers[str(t)]['constructive_rollout_witnesses']=witnesses
        layers[str(t)]['unwitnessed_search_states']=[i for i in range(4)if str(i)not in witnesses]
    dump(HERE/'FIVE_LAYER_FINAL.json',layers)
    failures=collections.Counter((r['type_id'],r['controller'],r['outcome'])for r in rows.values())
    dump(HERE/'FAILURE_TYPES.json',dict(counts=[dict(type_id=t,controller=c,outcome=o,n=n)for(t,c,o),n in failures.items()],interpretation='Timeout is not deadlock. Unwitnessed cases remain jointly uncertified. Solver exceptions and internal nonoptimal status counts are separate. Geometry-rejected forest has zero rollouts.'))
    # A complete compact, unique row index accompanies immutable detailed raw records.
    with(HERE/'candidates_unique.jsonl').open('w')as f:
        for uid,r in sorted(rows.items()):f.write(json.dumps({k:r.get(k)for k in('rollout_uid','scene','type_id','controller','state_index','solo','purpose','spec','success','collision','timeout','numerical_failure','outcome','completion_time','truncated_time','wall_seconds','trace','trace_sha256')},ensure_ascii=False)+'\n')
    recommended=recommendation or practical;fixed=families['families'][recommended['index']]['fixed']
    dimension_text='；'.join(f"{d}维：{f['enough_types']}类"for d,f in best_by_dimension.items())
    pertype='；'.join(f"{TITLES[t]} {recommended['by_type'][str(t)]['oracle']['success']}/8 对 {recommended['by_type'][str(t)]['fixed']['success']}/8"for t in range(1,18))
    text=[f"# 十七类场景上的修正基维度研究\n\n按预登记规则推荐：{('未找到满足 M−1 条件的五维以内族，以下另列有限池中实用最佳'if recommendation is None else str(recommended['dimension'])+'维')}，列为 {'、'.join(recommended['column_names'])}。逐类留出 Oracle@16 对固定参数成功数：{pertype}。覆盖缺口：{'、'.join(TITLES[t]for t in gaps)}。各维度最好族达到 6/8 的类型数：{dimension_text}。\n\n",
        '所有结论限于已冻结、已执行的有限候选池。森林场景在给定间距下几何不可实现，表中 0/8 是不可评估占位，不是控制失败。其余 16 类采用两个冻结名义控制器；每类仅 4 个独立物理初态，各控制器共享同一初态与随机流。\n\n',
        '## 场景、标定与五层证据\n\n[完整标定](CALIBRATION.json)、[冻结场景与状态](SCENES_FROZEN.json)、[最终五层证据](FIVE_LAYER_FINAL.json)、[几何与安全边距](GEOMETRY_CONSISTENCY.json)。单机能力门均按 ≥95% 判断；未满足难度与 A* 见证条件的类型按最近档保留，仍明确记录未认证。成功 rollout 可补充构造性联合可解见证；超时不能推出死锁或不可解。\n\n',
        '场景类型来源是几何与交通分类的移植，具体尺寸按本轮 prompt 新建，不是论文原地图复现。圆周、交叉流、同向流、门、盲角与集结对应 [Francis 等场景清单](https://arxiv.org/abs/2306.16740)；森林的动机对应 [BARN](https://www.cs.utexas.edu/~xiao/BARN/BARN.html)，本轮森林几何被拒绝，没有运行 BARN 官方地图。\n\n',
        '| 场景 | 冻结数量/半径 | 路线引导 | 安全基线成功 | A* 见证 |\n|---|---|---|---|---|\n']
    for e in frozen['entries']:
        if 'scene'not in e:text.append(f"|{e['title']}|几何不成立|—|—|—|\n");continue
        s=e['scene'];m=e['calibration'];text.append(f"|{e['title']}|{s['config']['num_agents']} / {s['config']['agent_radius']}m|{s['route_guidance']}|{m.get('safety_success')}/{m.get('safety_n')}|{m.get('witness_success')}/4|\n")
    text += ['\n森林排斥证据：每个 0.3m 方块向外膨胀 0.25m 后面积为 0.3²+4×0.3×0.25+π×0.25²；12 个的面积 7.03619m²，必须互不重叠且包含在 2.5m 方场地内，其面积仅 6.25m²。该矛盾与控制器无关。\n\n',
        '![冻结场景示意](scene_overview.svg)\n\n## 场景描述与筛选证据\n\n[描述量与距离矩阵](SCENE_DESCRIPTORS.json) 仅使用几何和初始路径。描述量不同不构成机制独立性的数学证明。[最小支持集](MINIMAL_SUPPORTS.json) 用全部成功见证的 18 列子集闭包求解；必需列仅相对于本轮有限池。S1、S2、S3、S4 的候选清单、preflight、执行清单均保存在同目录。低维补搜受 2,000 局上限约束，未找到成功不等于 Basis 不可能成功。\n\n',
        '## 留出结果与部署差距\n\n| 修正族 | 维度 | 达到6/8的类型 | Oracle覆盖 | 固定参数成功 | 固定成功平均完成时间(s) | 固定全队列截断均时(s) |\n|---|---|---|---|---|---|---|\n']
    for f in fs:text.append(f"|{' + '.join(f['column_names'])}|{f['dimension']}|{f['enough_types']}/17|{f['oracle']['success']}/{f['oracle']['n']}|{f['fixed']['success']}/{f['fixed']['n']}|{f['fixed']['success_mean_time']}|{f['fixed']['all_cohort_truncated_mean']}|\n")
    text += [f'\n推荐族固定参数：`{json.dumps(fixed,ensure_ascii=False)}`。完整逐场景结果见 [per_type_results.csv](per_type_results.csv)，预算曲线见 [budget_curves.csv](budget_curves.csv)。参数与族在读取留出结果前冻结。维度推荐使用预登记的留出决策规则，是本测试集上的描述性选择，不是另一次独立泛化确认。\n\n',
        '## 论文对应 baseline\n\n五篇论文的六个机制版本使用上一轮冻结参数和同一实际 SI 物理环境。它们保留论文自身 controller、安全 solver、优先级/临时目标机制，区别于两个冻结 nominal 下的修正列消融；不把不同控制模型的性能差当作单纯 Basis 表达能力差。论文原生 benchmark 与原理论假设未复现，原端口适配限制沿用 [上轮报告](../paper_baselines_20261028/report.md)。原始代码 commit 和依赖见 [端口来源](PAPER_PORT_INPUTS.json)。\n\n',
        '| 论文机制端口 | 成功 | 碰撞/超时/数值unknown | success条件平均时间(s) | 全队列截断均时(s) | 内部solver失败事件 |\n|---|---|---|---|---|---|\n']
    for method,v in paper_stats.items():
        s=v['overall'];text.append(f"|{method}|{s['success']}/{s['n']}|{s['collision']}/{s['timeout']}/{s['numerical_unknown']}|{s['success_mean_time']}|{s['all_cohort_truncated_mean']}|{s['solver_internal_failure_events']}|\n")
    text += ['\n经典右手目标偏转与群中心旋转启发式是本轮 prompt 指定的附加参照，结果见 [REFERENCE_RESULTS.json](REFERENCE_RESULTS.json)；它们不冒充 Wang、Chen 或 Merry-Go-Round 的完整论文方法，也不参与维度判定。\n\n',
        '## 时间、置信区间与计算成本\n\n成功条件均值只包含成功局；配对成功时间只在两方法均成功的同一 controller×state 上比较；失败局的截断时间使用完整任务时限。Oracle 完成时间取冻结顺序中第一个成功候选的时间，不把最短成功候选时间充作部署速度。Oracle 的计算成本应累加全部 16 候选，不能只算被选成功局。逐候选 wall_seconds 与完整轨迹均保留。Wilson 95% 区间是案例层描述区间，两个 nominal 的成败相关、每类仅四个独立物理初态，不能解释为八个独立样本的严格总体置信区间。\n\n',
        '## 正确性与限制\n\n[路线引导等价](ROUTE_EQUIVALENCE.json) 为 50 对完整局；[原三列集成回归](B3_EQUIVALENCE.json) 为两组非零历史参数、4 个初态×2个 nominal；[工程加速认证](ACCELERATION_CERTIFICATE.json) 明确列出启用项、200 对逐字节比较、随机键与稀疏数组检查。原安全 solver、重试、恢复、速度上限、物理 step 与成功判据沿用生产实现。T1 使用 prompt 的名称“目标右侧旋转”，但原代码正系数是逆时针 J，负系数才是顺时针；本轮保持该原定义。\n\n',
        '超时、碰撞、solver unknown 与内部非最优事件分别统计。某状态没有成功见证，只能说有限搜索未解决；尚不能区分搜索不足、时序不足、Basis 不足或联合不可解。原冻结 16 候选本轮不做阶段切换，不能据此排除时序收益。\n\n',
        '## 本轮未包括\n\n学习型规划器、Generator/Critic 训练、人机互动、非完整约束原生动力学、大地图、50 台以上、可选移动障碍与视频均未纳入。森林需用户修改几何规格后另开实验，不能在本轮偷偷缩小间距或方块。原始数据与 SQLite 保留本地，Git 提交小型代码和报告不等于大型轨迹已异地备份。\n']
    (HERE/'REPORT.md').write_text(''.join(text))
    try:
        subprocess.run(['/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python',str(HERE/'plot.py')],check=True)
    except Exception as exc:
        with(HERE/'GAPS.md').open('a')as f:f.write('\n场景图绘制未完成：'+repr(exc)+'\n')
    audit=final_integrity();executions={p.stem:read(p).get('complete')for p in HERE.glob('*_execution.json')}
    dump(HERE/'COMPLETE.json',dict(status='completed_finite_screening'if all(executions.values())else'partial_evidence_retained',all_planned_batches_complete=all(executions.values()),new_rollout_records=len(rows),attempts=len(list((HERE/'attempts').glob('*.json'))),geometry_types_evaluable=16,requested_types=17,verdict_sha256=sha(HERE/'VERDICT.json'),report_sha256=sha(HERE/'REPORT.md'),execution_status=executions,integrity_passed=True,scientific_scope='Finite Oracle@16, no continuous optimum or impossibility proof. Forest is a geometry gap.'))
    (HERE/'STATUS.md').write_text('# 当前状态\n\n本轮有限候选实验已交付，详见 REPORT.md 与 COMPLETE.json。森林规格几何不成立；其余16类的结果、数值unknown和未完成批次均按文件原值报告。\n')
    print('ANALYSIS DELIVERED',verdict['recommended_columns'],flush=True)

if __name__=='__main__':main()
