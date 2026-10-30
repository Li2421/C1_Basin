"""Physical-state paired reporting from immutable labels and completed plans."""
import guard
from guard import HERE
import json,csv,collections,datetime,math,hashlib,os
from fractions import Fraction
import numpy as np
from statistic_rules import hierarchical_interval as _hierarchical_interval,paired_signflip as _paired_signflip,holm,verdict
from method_names import NAMES

LEARNED='学习选修正参数'
INTERNAL={LEARNED,'只加安全投影','不加安全投影','经典目标右侧规则'}
_PORT_GATES=None
VARIANTS={'frozen':'冻结学习流水线/内部参照','global_tuned':'训练集调参','author_default':'作者默认·物理对齐',
          'author_original':'按原文安全距离','type_tuned':'按类型调参','mechanism_disabled':'去掉死锁机制',
          'first_point':'冻结首组','internal':'内部参照','legacy_geometry':'原墙体模型·补充',
          'consistent_sensitivity':'一致墙体模型·敏感性'}


def statistic_cache(kind,arguments,compute):
    """Reuse identical statistical computations, bound to exact algorithm SHA."""
    def encode(value):
        if isinstance(value,Fraction):return dict(fraction=[value.numerator,value.denominator])
        raise TypeError(type(value).__name__)
    source=hashlib.sha256((HERE/'statistic_rules.py').read_bytes()).hexdigest()
    descriptor=json.dumps(dict(kind=kind,algorithm_sha256=source,arguments=arguments),sort_keys=True,separators=(',',':'),default=encode)
    key=hashlib.sha256(descriptor.encode()).hexdigest();folder=HERE/'statistical_cache';folder.mkdir(exist_ok=True)
    path=folder/(key+'.json')
    if path.exists():return json.loads(path.read_text())['result']
    result=compute();temporary=folder/(key+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(dict(kind=kind,algorithm_sha256=source,result=result),ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)
    return result


def hierarchical_interval(blocks):return statistic_cache('hierarchical_interval',blocks,lambda:_hierarchical_interval(blocks))


def paired_signflip(values):return statistic_cache('paired_signflip',values,lambda:_paired_signflip(values))


def read(name):return json.loads((HERE/name).read_text())


def write(name,value):
    path=HERE/name;path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w')as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    temporary.replace(path)


def display(method):return NAMES.get(method,method)


def port_context(benchmark,method,names):
    global _PORT_GATES
    names=set(names)
    if method in INTERNAL or method.startswith(('ours_','internal_')):
        return dict(planned_scenes=len(names),applicable_scenes=None,port_status='冻结C1流水线/内部参照',port_problem_not_fully_resolved=None)
    if _PORT_GATES is None:_PORT_GATES=read('PORT_GATES.json')
    gates=_PORT_GATES
    warnings=sorted({r['benchmark']for r in gates['pilots']if r['method']==method and r['port_problem_not_fully_resolved']})
    applicable={r['scene']for r in gates['applicability']if r['benchmark']==benchmark and r['method']==method and r['applicable']is True}
    return dict(planned_scenes=len(names),applicable_scenes=len(names&applicable),
        port_status='端口问题未完全解决（'+','.join(warnings)+'训练试跑）'if warnings else'作者代码移植；原环境复现通过',
        port_problem_not_fully_resolved=bool(warnings))


def collect():
    split=read('SPLITS_FROZEN.json');variants=read('CONSISTENT_BOX_SCENES.json')
    from scene_catalog import catalog
    scenes=catalog('A',version='original');rows=[]
    timing={}
    if(HERE/'TIME_BACKFILL_RESULTS.json').exists():
        timing={tuple(r['identity']):r['result']for r in read('TIME_BACKFILL_RESULTS.json')['records']if r.get('result')}
    for filename,part in [('FROZEN_HISTORICAL_LABELS.json','test'),('FROZEN_VALIDATION_LABELS.json','validation')]:
        for old in read(filename)['records']:
            scene=scenes[old['scene']];category=scene['factor'];horizon=scene['config']['max_steps']*scene['config']['dt']
            variant='legacy_geometry'if variants['primary_by_category'].get(category,False)and old['scene']in variants['scene_records']else'frozen'
            outcome=old['outcome'];result=dict(success=old['success'],outcome=outcome,collision=int(outcome=='collision'),timeout=int(outcome=='timeout'),
                numerical_failure=int(outcome=='unknown'),completion_time=None,truncated_time=horizon if not old['success']else None,horizon_time=horizon)
            identity=(old['scene'],old['state'],old['name'],old['method'])
            replay=timing.get(identity)
            if replay and replay['success']==old['success']and replay['outcome']==outcome:
                for key in ('completion_time','truncated_time','trace','wall_seconds'):
                    if key in replay:result[key]=replay[key]
            rows.append(dict(benchmark='A',part=part,scene=old['scene'],category=category,state=old['state'],nominal=old['name'],
                             method=old['method'],variant=variant,result=result,historical_authoritative=True))
    if(HERE/'CONSISTENT_OURS_RESULTS.json').exists():
        for row in read('CONSISTENT_OURS_RESULTS.json')['records']:
            row=dict(row)
            row['variant']='frozen'if variants['primary_by_category'].get(row['category'],False)else'consistent_sensitivity'
            rows.append(row)
    for path in sorted((HERE/'test_tables').glob('test_*.json')):
        rows.extend(json.loads(path.read_text())['rows'])
    return rows


def physical(rows,expected_nominals=None):
    groups=collections.defaultdict(list)
    for row in rows:groups[(row['scene'],row['state'])].append(row)
    values={};missing=[]
    for key,group in groups.items():
        expected=6 if group[0]['benchmark']=='A'and group[0]['method']in INTERNAL else 2 if group[0]['benchmark']=='B'and group[0]['method'].startswith(('ours_','internal_'))else 1
        if expected_nominals is not None:expected=expected_nominals
        if len(group)!=expected or any(r.get('result')is None for r in group):missing.append(key);continue
        assert len({r['nominal']for r in group})==len(group),'Duplicate nominal evidence'
        values[key]=dict(category=group[0]['category'],success=sum(Fraction(int(r['result']['success']))for r in group)/expected,
                         collision=sum(Fraction(int(r['result']['collision']))for r in group)/expected,
                         timeout=sum(Fraction(int(r['result']['timeout']))for r in group)/expected,
                         unhandled=sum(Fraction(int(bool(r['result'].get('paper_original',{}).get('numerical_unknown',r['result'].get('numerical_failure',0)))))for r in group)/expected,
                         episodes=group)
    return values,missing


def blocks(values,key):
    out={}
    for (scene,state),value in sorted(values.items()):out.setdefault(str(value['category']),{}).setdefault(scene,[]).append(float(value[key]))
    return out


def summary(rows,interval=True,expected_nominals=None):
    values,missing=physical(rows,expected_nominals);known=[r['result']for r in rows if r.get('result')is not None]
    success=[r for r in known if r['success']];times=[r['completion_time']for r in success if r.get('completion_time')is not None]
    capped=[r['truncated_time']for r in known if r.get('truncated_time')is not None]
    return dict(physical_states=len(values),missing_physical_states=len(missing),episodes=len(rows),observed_episodes=len(known),
        success_rate=float(sum((v['success']for v in values.values()),Fraction())/len(values))if values else None,
        success_interval=hierarchical_interval(blocks(values,'success'))if interval else None,
        collision_rate=float(sum(v['collision']for v in values.values())/len(values))if values else None,
        timeout_rate=float(sum(v['timeout']for v in values.values())/len(values))if values else None,
        numerical_unknown_episodes=sum(int(r.get('numerical_failure',0))for r in known),
        unhandled_episodes=sum(bool(r.get('paper_original',{}).get('numerical_unknown',r.get('numerical_failure',0)))for r in known),
        success_episodes=len(success),successful_times_observed=len(times),
        success_conditional_seconds=float(np.mean(times))if times and len(times)==len(success)else None,
        observed_success_time_subset_seconds=float(np.mean(times))if times else None,
        all_queue_capped_seconds=float(np.mean(capped))if capped and len(capped)==len(known)and len(known)==len(rows)else None,
        missing_success_times=len(success)-len(times),missing_result_uids=[r.get('query_uid')for r in rows if r.get('result')is None])


def paired_time(ours,paper):
    a=collections.defaultdict(list);b={}
    for row in ours:a[(row['scene'],row['state'])].append(row.get('result'))
    for row in paper:b[(row['scene'],row['state'])]=row.get('result')
    differences=[];pairs=0;missing=0
    for key,values in a.items():
        other=b.get(key)
        if not other or not other['success']:continue
        good=[r for r in values if r and r['success']]
        if not good:continue
        if other.get('completion_time')is None or any(r.get('completion_time')is None for r in good):missing+=1;continue
        differences.append(float(np.mean([r['completion_time']-other['completion_time']for r in good])));pairs+=len(good)
    return dict(mean_ours_minus_paper_seconds=float(np.mean(differences))if differences else None,physical_states=len(differences),nominal_pairs=pairs,missing_time_physical_states=missing,
                unit='Mean over successful nominal pairs inside physical state, then equal mean over physical states')


def comparisons(rows):
    ours=[r for r in rows if r['benchmark']=='A'and r['part']=='test'and r['method']==LEARNED and r['variant']=='frozen']
    own,_=physical(ours);out={};pvalues={}
    native=read('NATIVE_REPRODUCTION_RESULTS.json')['methods']
    for method in native:
        paper=[r for r in rows if r['benchmark']=='A'and r['part']=='test'and r['method']==method and r['variant']=='global_tuned']
        if not native[method]['reproducible']:
            pvalues[method]=None;out[method]=dict(excluded=True,reason=native[method].get('reason'));continue
        other,_=physical(paper);keys=sorted(own.keys()&other.keys());grouped={};scenes=collections.defaultdict(list)
        for key in keys:
            difference=own[key]['success']-other[key]['success'];scene,state=key
            grouped.setdefault(str(own[key]['category']),{}).setdefault(scene,[]).append(float(difference));scenes[scene].append(difference)
        interval=hierarchical_interval(grouped);test=paired_signflip([sum(v,Fraction())/len(v)for v in scenes.values()]);pvalues[method]=test['p']
        out[method]=dict(excluded=False,paired_difference=interval,signflip=test,paired_time=paired_time(ours,paper),
                         paired_physical_states=len(keys),preregistered_physical_states=384)
    adjusted=holm(pvalues)
    for method,row in out.items():
        row['holm_p']=adjusted[method]
        if not row['excluded']:row['verdict']=verdict(adjusted[method],row['paired_difference']['ci95'])
    return dict(methods=out,holm_family_size=3,excluded_tests_have_no_fabricated_p=True)


def missing_sensitivity(rows,grouped):
    """Separate prespecified sensitivity; never mutate primary evidence."""
    def values(group):
        by_state=collections.defaultdict(list)
        for row in group:by_state[(row['scene'],row['state'])].append(row)
        result={};missing_episodes=0
        for key,episodes in by_state.items():
            first=episodes[0]
            expected=6 if first['benchmark']=='A'and first['method']in INTERNAL else 2 if first['benchmark']=='B'and first['method'].startswith(('ours_','internal_'))else 1
            assert len(episodes)<=expected,('Duplicate evidence in missingness sensitivity',key)
            assert len({r['nominal']for r in episodes})==len(episodes)
            known=[r['result']for r in episodes if r.get('result')is not None]
            missing_episodes+=expected-len(known)
            result[key]=dict(category=first['category'],success=Fraction(sum(int(r['success'])for r in known),expected))
        return result,missing_episodes
    tables=[]
    for (benchmark,part,method,variant),group in sorted(grouped.items()):
        observed,missing=values(group)
        tables.append(dict(benchmark=benchmark,part=part,method=display(method),variant=VARIANTS.get(variant,variant),
            physical_states=len(observed),imputed_missing_episodes=missing,
            success_rate=float(sum((v['success']for v in observed.values()),Fraction())/len(observed))if observed else None,
            success_interval=hierarchical_interval(blocks(observed,'success'))))
    own,_=values(grouped.get(('A','test',LEARNED,'frozen'),[]));paired={};pvalues={}
    for method,native in read('NATIVE_REPRODUCTION_RESULTS.json')['methods'].items():
        if not native['reproducible']:
            paired[method]=dict(excluded=True,reason='Native reproduction gate; no fabricated missing-as-failure test.');pvalues[method]=None;continue
        other,_=values(grouped.get(('A','test',method,'global_tuned'),[]));blocks_by_category={};by_scene=collections.defaultdict(list)
        keys=sorted(own.keys()&other.keys())
        for key in keys:
            scene,state=key;delta=own[key]['success']-other[key]['success']
            blocks_by_category.setdefault(str(own[key]['category']),{}).setdefault(scene,[]).append(float(delta));by_scene[scene].append(delta)
        test=paired_signflip([sum(v,Fraction())/len(v)for v in by_scene.values()]);pvalues[method]=test['p']
        paired[method]=dict(excluded=False,paired_physical_states=len(keys),paired_difference=hierarchical_interval(blocks_by_category),signflip=test)
    adjusted=holm(pvalues)
    for method,row in paired.items():row['holm_p']=adjusted[method]
    write('MISSING_AS_FAILURE_SENSITIVITY.json',dict(rows=tables,primary_comparisons=paired,holm_family_size=3,
        interpretation='Prespecified sensitivity only: unavailable nominal results contribute zero success within the original physical block. No collision or time values are invented. Withdrawn/native-excluded methods remain excluded.'))


def write_csv(name,rows):
    if not rows:return
    with(HERE/name).open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def cpu_summary(rows):
    arrays=[];seen=set();episodes=0;missing=0;setup=[];dictionary=[]
    for row in rows:
        result=row.get('result')
        if not result:continue
        uid=result.get('rollout_uid',result.get('trace'))
        if uid is None:missing+=1;continue
        if uid in seen:continue
        seen.add(uid);trace=result.get('trace')
        if not trace or not __import__('pathlib').Path(trace).exists():missing+=1;continue
        with np.load(trace)as saved:
            if 'step_seconds' in saved:arrays.append(np.asarray(saved['step_seconds']));episodes+=1
            else:missing+=1
        if result.get('setup_seconds')is not None:setup.append(result['setup_seconds'])
        if 'dictionary'in result and 'seconds'in result['dictionary']:dictionary.append(result['dictionary']['seconds'])
    values=np.concatenate(arrays)if arrays else np.array([])
    return dict(observed_episodes=episodes,missing_trace_episodes=missing,control_steps=len(values),
        per_step_median_seconds=float(np.median(values))if len(values)else None,
        per_step_p95_seconds=float(np.quantile(values,.95))if len(values)else None,
        observed_setup_mean_seconds=float(np.mean(setup))if setup else None,
        observed_dictionary_total_seconds=float(np.sum(dictionary))if dictionary else None,
        generator_critic_startup_seconds=None,
        interpretation='Recorded monotonic wall time on a single CPU for the whole control step, including nominal/safety/correction; not isolated correction-only incremental cost. Dictionary construction is only the separately logged component. Missing historical measurements are not zero; setup/import and Generator/Critic evidence are separate; see GENERATOR_CRITIC_TIMING_AUDIT.json.')


def difference(first,second):
    a,_=physical(first);b,_=physical(second);grouped={}
    for key in sorted(a.keys()&b.keys()):
        scene,state=key;grouped.setdefault(str(a[key]['category']),{}).setdefault(scene,[]).append(float(a[key]['success']-b[key]['success']))
    return hierarchical_interval(grouped)


def supplementary(rows,grouped):
    cpu=[];applicable=[];gains=[];matched=[]
    gates=read('PORT_GATES.json')['applicability']
    for key,group in sorted(grouped.items()):
        benchmark,part,method,variant=key
        cpu.append(dict(benchmark=benchmark,part=part,method=display(method),variant=VARIANTS.get(variant,variant),**cpu_summary(group)))
        if variant=='global_tuned'and method in NAMES and not method.startswith('ours_'):
            names={r['scene']for r in gates if r['benchmark']==benchmark and r['method']==method and r['applicable']is True}
            filtered=[r for r in group if r['scene']in names]
            applicable.append(dict(benchmark=benchmark,part=part,method=display(method),applicable_scenes=len(names&{r['scene']for r in group}),**summary(filtered)))
            disabled=grouped.get((benchmark,part,method,'mechanism_disabled'),[])
            gains.append(dict(benchmark=benchmark,part=part,method=display(method),gain='trained minus mechanism-disabled',
                              interval=difference(group,disabled),descriptive_only=True))
        if benchmark=='A'and part=='test'and variant in('global_tuned','author_default','author_original','type_tuned'):
            common=[r for r in group if 8<=r['state']<=15]
            matched.append(dict(method=display(method),variant=VARIANTS.get(variant,variant),matched_states='8–15',**summary(common)))
    for part in ('test','validation'):
        own=grouped.get(('A',part,LEARNED,'frozen'),[]);safe=grouped.get(('A',part,'只加安全投影','frozen'),[])
        gains.append(dict(benchmark='A',part=part,method=LEARNED,gain='frozen correction minus safety-only',interval=difference(own,safe),descriptive_only=True))
    for method in ('ours_main','ours_four'):
        own=grouped.get(('B','test',method,'global_tuned'),[]);safe=grouped.get(('B','test','internal_safety','internal'),[])
        gains.append(dict(benchmark='B',part='test',method=display(method),gain='trained correction minus safety-only',interval=difference(own,safe),descriptive_only=True))
    write('COMPUTATION_TABLE.json',dict(rows=cpu));write('APPLICABLE_ONLY.json',dict(rows=applicable))
    write('MECHANISM_GAINS.json',dict(rows=gains));write('DEFAULT_TUNED_MATCHED_STATES.json',dict(rows=matched))


def opening(stats,paired,gate):
    def pct(value):return '缺失'if value is None else f'{100*value:.1f}%'
    active=[m for m,v in paired['methods'].items()if not v['excluded']]
    pieces=['作者原环境复现后保留Zhang T-RO 2025 死锁感知控制与Lee IROS 2025 旋转木马；Chen RA-L 2023 动态优先级＋右手规则＋安全走廊因原生轨迹帧间连续碰撞退出，另三个无作者代码/参数的方法退出。']
    for method in active:
        applicable=[r for r in gate['applicability']if r['benchmark']=='A'and r['method']==method]
        row=next((r for r in stats if(r['benchmark'],r['part'],r['method'],r['variant'])==('A','test',method,'global_tuned')),None)
        if row:
            error=row['unhandled_episodes']/row['observed_episodes']if row['observed_episodes']else None
            pieces.append(f"{display(method)}修复后正式测试未处理异常率{pct(error)}，单机适用场景{sum(r['applicable']is True for r in applicable)}/{len(applicable)}；")
        ci=paired['methods'][method]['paired_difference']
        pieces.append(f"主终点我们减该方法{pct(ci['mean'])}，95%区间{ci['ci95']}，{paired['methods'][method]['verdict']}。")
    matched=read('DEFAULT_TUNED_MATCHED_STATES.json')['rows']
    ranks={}
    for variant in ('训练集调参','作者默认·物理对齐'):
        group=[r for r in matched if r['variant']==variant and r['success_rate']is not None]
        group.sort(key=lambda r:(-r['success_rate'],r['method']))
        ranks[variant]=[(r['method'],r['success_rate'])for r in group]
    pieces.append('共同状态8–15的默认/调参成功率：'+str(ranks)+'；完整并列关系见匹配表。')
    breadth=[r for r in stats if r['benchmark']=='B'and r['part']=='test'and r['variant']=='global_tuned']
    pieces.append('基准乙固定参数成功率：'+'，'.join(r['method_display']+' '+pct(r['success_rate'])for r in breadth)+'。')
    pieces.append('Oracle@16'+('已执行，有限池覆盖与首次成功组数见后表。'if(HERE/'ORACLE16_RESULTS.json').exists()else'尚未执行（可选，全部主线之后）。'))
    pieces.append('一致墙体版本方形与十字验证/测试初态合法比例均100%，按预登记规则进主表；学习模型训练于旧墙体模型。原代表场景参与过机制设计，验证集已被使用，共同环境端口不能冒充原论文直接复现。单机门及窄通道对齐前后变化见PORT_FIX_BEFORE_AFTER.json与NARROW_PASSAGE_DIAGNOSTIC.json。')
    return ''.join(pieces)


def run():
    analysis=read('ANALYSIS_PLAN_FROZEN.json')
    assert hashlib.sha256((HERE/'SCENE_NAMES.json').read_bytes()).hexdigest()==analysis['scene_names_sha256'],'HARD_STOP: scene display mapping changed after analysis freeze'
    for name,expected in analysis['analysis_sources_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,('HARD_STOP: frozen analysis source changed',name)
    rows=collect();grouped=collections.defaultdict(list)
    for row in rows:grouped[(row['benchmark'],row['part'],row['method'],row['variant'])].append(row)
    stats=[]
    for (benchmark,part,method,variant),group in sorted(grouped.items()):
        timing=cpu_summary(group)
        stats.append(dict(benchmark=benchmark,part=part,method=method,method_display=display(method),variant=variant,variant_display=VARIANTS.get(variant,variant),
                          cpu_step_median_seconds=timing['per_step_median_seconds'],cpu_step_p95_seconds=timing['per_step_p95_seconds'],
                          cpu_observed_episodes=timing['observed_episodes'],cpu_missing_trace_episodes=timing['missing_trace_episodes'],
                          **port_context(benchmark,method,{r['scene']for r in group}),**summary(group)))
    paired=comparisons(rows)
    write('SUMMARY_TABLES.json',dict(rows=stats));write('PRIMARY_PAIRED_DIFFERENCES.json',paired)
    flat=[]
    for row in stats:
        flat.append({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in row.items()})
    write_csv('MAIN_AND_SUPPLEMENT_TABLES.csv',flat)
    by_category=[];by_nominal=[]
    scene_names=read('SCENE_NAMES.json')
    category_names={'plus':'十字路口（各路宽）','box':'拥挤方形场地（各半径与密度）','obst':'带障碍圆圈对穿','swap':'开阔圆圈对穿'}
    for (benchmark,part,method,variant),group in sorted(grouped.items()):
        for category in sorted({str(r['category'])for r in group}):
            subset=[r for r in group if str(r['category'])==category]
            category_display=category_names.get(category,category)if benchmark=='A'else'；'.join(sorted({scene_names[r['scene']]['display']for r in subset}))
            by_category.append(dict(benchmark=benchmark,part=part,method=display(method),variant=VARIANTS.get(variant,variant),category=category,
                category_display=category_display,
                **port_context(benchmark,method,{r['scene']for r in subset}),**summary(subset,interval=True)))
        if benchmark=='A'and method in INTERNAL:
            for nominal in sorted({r['nominal']for r in group}):
                episode_rows=[r for r in group if r['nominal']==nominal]
                timing=cpu_summary(episode_rows)
                by_nominal.append(dict(part=part,method=method,variant=VARIANTS.get(variant,variant),nominal=nominal,
                    successes=sum(r['result']['success']for r in episode_rows if r.get('result')),
                    cpu_step_median_seconds=timing['per_step_median_seconds'],cpu_step_p95_seconds=timing['per_step_p95_seconds'],
                    **summary(episode_rows,expected_nominals=1)))
    write('CATEGORY_TABLES.json',dict(rows=by_category));write_csv('PER_NOMINAL.csv',by_nominal)
    write_csv('CATEGORY_TABLES.csv',[{k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in row.items()}for row in by_category])
    write('NORMALIZED_EVIDENCE.json',dict(rows=rows))
    supplementary(rows,grouped)
    missing_sensitivity(rows,grouped)
    from classical_appendix import run as classical_appendix
    classical_appendix()
    native=read('NATIVE_REPRODUCTION_RESULTS.json');consistent=read('CONSISTENT_BOX_SCENES.json');gate=read('PORT_GATES.json')
    lines=['# 论文基线正式比较','',
        opening(stats,paired,gate),
        '', '## 基准甲主表','',
        '| 方法 | 口径 | 物理状态 | 成功率与95%区间 | 碰撞/超时率 | 未处理异常 | 单机适用/场景 | 端口状态 | 成功条件均时·秒 | 全队列截断均时·秒 | 单步中位/p95·毫秒 | 缺完成时间的成功回合 |',
        '|---|---|---:|---|---|---:|---|---|---:|---:|---:|---:|']
    def number(x):return '缺失'if x is None else f'{x:.4f}'
    for r in stats:
        if r['benchmark']=='A'and r['part']=='test'and r['variant']not in('legacy_geometry','consistent_sensitivity'):
            cpu='/'.join(number(1000*r[k])if r[k]is not None else'缺失'for k in ('cpu_step_median_seconds','cpu_step_p95_seconds'))
            applicability='未定义'if r['applicable_scenes']is None else str(r['applicable_scenes'])+'/'+str(r['planned_scenes'])
            lines.append(f"| {r['method_display']} | {r['variant_display']} | {r['physical_states']} | {number(r['success_rate'])} {r['success_interval']['ci95']} | {number(r['collision_rate'])}/{number(r['timeout_rate'])} | {r['unhandled_episodes']} | {applicability} | {r['port_status']} | {number(r['success_conditional_seconds'])} | {number(r['all_queue_capped_seconds'])} | {cpu} | {r['missing_success_times']} |")
    lines+=['','## 主终点配对差','', '| 方法 | 我们减外部成功率差 | 95%区间 | Holm校正p | 判读 |','|---|---:|---|---:|---|']
    for method,r in paired['methods'].items():
        if r['excluded']:lines.append(f'| {display(method)} | 未检验 | 原环境复现未通过 | 缺失 | 退出 |');continue
        ci=r['paired_difference'];lines.append(f"| {display(method)} | {number(ci['mean'])} | {ci['ci95']} | {number(r['holm_p'])} | {r['verdict']} |")
    lines+=['','## 基准乙广度表','',
        '| 方法 | 口径 | 物理状态 | 成功率与95%区间 | 未处理异常 | 端口状态 | 成功条件均时·秒 | 全队列截断均时·秒 |','|---|---|---:|---|---:|---|---:|---:|']
    for r in stats:
        if r['benchmark']=='B':lines.append(f"| {r['method_display']} | {r['variant_display']} | {r['physical_states']} | {number(r['success_rate'])} {r['success_interval']['ci95']} | {r['unhandled_episodes']} | {r['port_status']} | {number(r['success_conditional_seconds'])} | {number(r['all_queue_capped_seconds'])} |")
    breadth=[r for r in by_category if r['benchmark']=='B'and r['part']=='test'and r['variant']==VARIANTS['global_tuned']]
    methods=[display(m)for m in ('ours_main','ours_four','dac_author_si','mgr_author_unicycle_tracking')]
    cells={(r['category'],r['method']):r for r in breadth}
    lines+=['','## 十七类逐类固定参数对照','',
        '每格为成功率（物理初态数；成功条件均时·秒）；置信区间、缺失与其余口径详见CATEGORY_TABLES.csv。所有外部方法的端口状态同上表。','',
        '| 场景与几何 | '+' | '.join(methods)+' |','|---|'+'---:|'*len(methods)]
    for category in sorted({r['category']for r in breadth},key=int):
        label=next(r['category_display']for r in breadth if r['category']==category);values=[]
        for method in methods:
            row=cells.get((category,method))
            values.append('缺失'if row is None else f"{number(row['success_rate'])}（{row['physical_states']}；{number(row['success_conditional_seconds'])}）")
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    if(HERE/'ORACLE16_RESULTS.json').exists():
        oracle=read('ORACLE16_RESULTS.json');lines+=['','## 可选有限池Oracle@16','',
            '| 方法 | 完整搜索 | 成功/案例 | 成功首次组数均值 | 全案例搜索组数均值 |','|---|---|---:|---:|---:|']
        for result in oracle['methods']:
            s=result.get('summary',{});lines.append(f"| {display(result['method'])} | {result.get('complete')} | {s.get('successes','—')}/{s.get('cases','—')} | {s.get('first_success_mean','—')} | {s.get('all_case_logical_search_mean','—')} |")
        lines+=['','C1每个名义控制器分别搜索，外部方法每个物理初态搜索一次；完整搜索的总体成功率与区间按物理状态折叠。未完成搜索的success/cases只代表已验证覆盖下界，未搜索完单列，不当作有限池失败；不输出冒充全队列的已解决子集CI。训练选中的17–32组可能不在Oracle@16前缀内，ORACLE_SUBSET_CHECK.json明确核验保证适用范围。预算曲线、逐候选η/参数、数值未知、作者原生异常和首次成功轨迹保存在ORACLE16_RESULTS.json及oracles/；大日志保留在SHA绑定原始文件。']
    lines+=['','验证集、基准乙、按类型、按原文距离、机制关闭与Oracle只作描述；完整各类分母、缺失与原始记录见JSON/CSV。主终点按物理状态配对，学习规划器作为同一物理块平均；场景类别内重抽场景、场景内重抽状态10000次，场景符号翻转100000次，三项预登记比较做Holm校正。缺失按失败计的预登记敏感性另见MISSING_AS_FAILURE_SENSITIVITY.json，不改主表。七个经典控制器的旧证据分表见CLASSICAL_CONTROLLER_APPENDIX.csv：沿用历史墙体模型，仅作历史描述，不与新物理一致主表直接排名。',
        '', '## 原文目标几何假设诊断','',
        'Lee IROS 2025 旋转木马原文III-E对目标与障碍、目标之间的距离还有额外充分假设。只按目标几何作数值检查，基准乙136个测试物理初态中，安全倍率1.0时61个、原文1.1时53个满足所检查条款；其余不能据此判物理不可解或算法必败。原文目标距障碍条款与作者代码实际CBF阈值分别记录，不混为一条约束。主表和单机适用性表都没有据此删除任何案例；低训练成绩不被直接解释为论文死锁机制弱。数据、原文页码和方法见MGR_DOMAIN_AUDIT.json及[原文III-E](https://arxiv.org/pdf/2503.05848v1#page=3)。',
        '', '## 计算与时间限制','',
        '成功条件均时仅在全部成功回合的时间已知时给出；否则记缺失并另存已知时间子集均值，不能拿子集冒充全队列。非成功回合按冻结时限计入截断时间，不用失败提前终止时间替代；共同成功的配对时间见PRIMARY_PAIRED_DIFFERENCES.json。历史生成器/critic一次性计算时间未记录，冻结参数查表不被当作已测零开销。',
        '', '## 局限','',
        '代表17场景参与过通道项设计，存在场景层面设计泄漏；新测试初态不消除该限制。基准乙每类只有一张测试地图，区间只描述这些固定地图内初态变异，不能作为地图总体不确定度。验证集参与过原学习方法设计，只描述。较少自由参数的Oracle结果不等于线性基秩最优，也不等于部署性能。外部原生算法与共同环境端口的信息、动力学适配和适用性不同，逐项见INFORMATION_PARITY.md与PORT_FIXES.md。论文图号缺口、连续与离散安全差异、旧时间缺失和实际流程问题均保留在GAPS.md。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    required=[f'test_A_{v}.json'for v in ('global_tuned','author_default','author_original','type_tuned','mechanism_disabled')]
    required += [f'test_B_{v}.json'for v in ('global_tuned','author_default','author_original','type_tuned','mechanism_disabled','first_point','internal')]
    main_complete=all((HERE/'test_tables'/name).exists()and json.loads((HERE/'test_tables'/name).read_text())['observed_logical_rows']==json.loads((HERE/'test_tables'/name).read_text())['planned_logical_rows']for name in required)
    main_complete=main_complete and read('CONSISTENT_OURS_RESULTS.json')['complete']
    selections=read('PARAMETERS_FROZEN.json')['benchmarks']
    main_complete=main_complete and all(v.get('available')for values in selections.values()for v in values.values())
    main_complete=main_complete and all(v.get('available')for v in read('OURS_FIXED_FROZEN.json')['benchmark_B'].values())
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),main_numeric_report_saved=True,main_fixed_tests_complete=main_complete,
        summary_tables='SUMMARY_TABLES.json',primary='PRIMARY_PAIRED_DIFFERENCES.json',
        new_attempts=read('BUDGET.json')['actual_attempts'],remaining_deliverables=['Optional Oracle','Final integrity and process audit'])
    write('RESULTS.json',result)
    write('MAINLINE_READY_FOR_OPTIONAL.json',dict(complete=main_complete,required_tables=required))
    return result
