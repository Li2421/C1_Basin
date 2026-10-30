"""Read-only evidence audit of within-case eta sensitivity; no rollout or fitting."""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import json
import statistics

HERE=Path(__file__).resolve().parent
ART=HERE/'eta_learning_value_artifacts'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def key(spec):return json.dumps(spec,sort_keys=True,separators=(',',':'))


def run():
    assert Path('/mnt/ssd').is_mount() and ART.is_symlink()
    assert ART.resolve()==Path('/mnt/ssd/zhihan/Basin_C1/diagnostics/paper_baseline_protocol_v1/eta_learning_value_artifacts')
    snapshot=read(HERE/'ETA_LEARNING_VALUE_INPUTS.json')
    for row in snapshot['inputs']:assert sha(row['copy'])==row['sha256']
    source=ART/'inputs'
    complete=read(source/'COMPLETE.json')
    assert complete['required_numeric_work_complete']
    code=complete['minimum']['code']
    assert code=='level3_01'
    family=next(f for f in read(source/'COMBINATIONS.json')['combinations']if f['code']==code)
    assert family['free']==['G','R','T'] and family['bindings']=={'Q':'T','C':'R'}
    pool=family['candidate_pool'];assert len(pool)==64
    lookup={key(spec):i+1 for i,spec in enumerate(pool)};assert len(lookup)==64
    state=read(source/'SEARCH_level3_01_screen.json')
    groups=state['groups'];assert len(groups)==272
    identities={tuple(g['identity'])for g in groups};assert len(identities)==272
    assert all(g['identity'][3]in range(8)for g in groups)
    assert len({(g['scene'],g['identity'][3])for g in groups})==136
    observations={};case_records=[];first=collections.Counter();unknown=0
    for group in groups:
        identity=tuple(group['identity']);queries=group['queries']
        assert [q['index']for q in queries]==list(range(1,len(queries)+1))
        if group['success']:
            assert group['first_success']==len(queries)
            assert all(not q['result']['success']for q in queries[:-1]) and queries[-1]['result']['success']
            first[group['first_success']]+=1
        else:assert len(queries)==64 and not any(q['result']['success']for q in queries)
        for query in queries:
            result=query['result'];index=query['index']
            assert key(result['spec'])==key(pool[index-1])
            observations[(identity,index)]=dict(result=result,source='oracle_prefix',query_uid=query['query_uid'])
            unknown+=int(bool(result['numerical_failure']))
        case_records.append(dict(identity=list(identity),scene=group['scene'],oracle_success=bool(group['success']),
            first_success=group['first_success'],prefix_observed=len(queries)))
    # Existing complete +/-sign contrast supplies outcomes after early success.
    # Reuse only. This is not a new fixed-policy selection/coverage experiment.
    contrasts=[]
    fixed=[]
    for name in ('FIXED_level3_01_screen.json','FIXED_level3_01_screen_plus.json'):
        selected={tuple(r['identity']):r for r in read(source/name)['records']if r['identity'][3]in range(8)}
        assert set(selected)==identities
        fixed.append(selected)
        for identity,record in selected.items():
            index=lookup[key(record['spec'])];result=record['result']
            assert key(result['spec'])==key(record['spec'])
            prior=observations.get((identity,index))
            if prior:
                for field in ('success','outcome','completion_time'):
                    assert prior['result'][field]==result[field],('Conflicting old evidence',identity,index,field)
            else:observations[(identity,index)]=dict(result=result,source=name,query_uid=record['query_uid'])
    negative,positive=fixed
    negative_requests={r['rollout_uid']:r for r in read(source/'fixed_level3_01_screen_all_requests.json')['exact_configs']}
    positive_requests={r['rollout_uid']:r for r in read(source/'fixed_level3_01_screen_plus_all_requests.json')['exact_configs']}
    cells=collections.Counter();paired_times=[];by_stratum=collections.defaultdict(set);by_type=collections.defaultdict(collections.Counter)
    for identity in sorted(identities):
        a,b=negative[identity],positive[identity];ra,rb=a['result'],b['result']
        assert a['spec']==family['fixed'] and b['spec']==family['fixed_plus']
        sa,sb=a['spec'],b['spec']
        assert sa['eta']==sb['eta'] and sa['one_way']==sb['one_way']
        assert sa['coefficients']['C2']==sb['coefficients']['C2']
        assert sa['coefficients']['T1']==-.75 and sb['coefficients']['T1']==.75
        for field in ('state_uid','scenario_uid','seed_key','seed','initial_positions','initial_velocities','goals','controller','horizon_time'):
            assert ra[field]==rb[field],('Unmatched contrast',identity,field)
        na,nb=negative_requests[a['query_uid']],positive_requests[b['query_uid']]
        for field in ('state','scenario_payload','state_uid','rng_state_uid','seed_key','seed'):
            assert na[field]==nb[field],('Unmatched physical/PRNG root',identity,field)
        assert {k:v for k,v in na['controller_payload'].items()if k!='dictionary'}=={k:v for k,v in nb['controller_payload'].items()if k!='dictionary'}
        outcome=(int(ra['success']),int(rb['success']));name=f'{outcome[0]}{outcome[1]}'
        cells[name]+=1;by_type[identity[0]][name]+=1
        if outcome in ((1,0),(0,1)):by_stratum[(identity[0],identity[2])].add(name)
        if outcome==(1,1):paired_times.append(ra['completion_time']-rb['completion_time'])
        contrasts.append(dict(identity=list(identity),negative_success=outcome[0],positive_success=outcome[1],
            negative_outcome=ra['outcome'],positive_outcome=rb['outcome'],
            negative_seconds=ra['completion_time'],positive_seconds=rb['completion_time'],
            negative_uid=a['query_uid'],positive_uid=b['query_uid'],state_uid=ra['state_uid']))
    mixed=[];multiple=[]
    for record in case_records:
        identity=tuple(record['identity']);values=[v for(k,i),v in observations.items()if k==identity]
        successes=sum(bool(v['result']['success'])for v in values)
        failures=sum(not v['result']['success']and not v['result']['numerical_failure']for v in values)
        record.update(observed_candidates=len(values),observed_successful_candidates=successes,observed_failed_candidates=failures)
        if successes and failures:mixed.append(identity)
        if successes>=2:multiple.append(identity)
    matrix=ART/'case_eta_observed.csv'
    with matrix.open('w',newline='')as stream:
        fields=['type_id','level','nominal','state','candidate','observed','success','outcome','completion_seconds','goal','separation_clearance','rotation','admission_strength','query_uid']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for identity in sorted(identities):
            for index,spec in enumerate(pool,1):
                value=observations.get((identity,index));result=value['result']if value else None
                writer.writerow(dict(type_id=identity[0],level=identity[1],nominal=identity[2],state=identity[3],candidate=index,
                    observed=int(value is not None),success=result['success']if result else '',outcome=result['outcome']if result else '',
                    completion_seconds=result['completion_time']if result and result['success']else '',goal=spec['eta'][0],
                    separation_clearance=spec['eta'][2],rotation=spec['coefficients'].get('T1',0),
                    admission_strength=spec['one_way']['c1'],query_uid=value['query_uid']if value else ''))
    write(ART/'case_records.json',case_records);write(ART/'controlled_sign_contrasts.json',contrasts)
    differences=[abs(x)for x in paired_times]
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),family_code=code,family_name=family['name'],
        dimension=3,free=family['free'],bindings=family['bindings'],cases=272,physical_states=136,oracle_success=sum(first.values()),
        oracle_queries=sum(len(g['queries'])for g in groups),known_case_candidate_cells=len(observations),possible_cells=272*64,
        unobserved_cells=272*64-len(observations),numerical_unknown_prefix_queries=unknown,
        first_success_histogram=dict(sorted(first.items())),
        success_after_at_least_one_failed_eta=sum(v for k,v in first.items()if k>1),
        first_success_mean=statistics.mean([k for k,v in first.items()for _ in range(v)]),
        first_success_median=statistics.median([k for k,v in first.items()for _ in range(v)]),
        successful_candidate_indices_observed_as_first=len(first),
        cases_with_verified_success_failure_eta_contrast=len(mixed),cases_with_multiple_verified_successful_etas=len(multiple),
        controlled_sign=dict(negative_spec=family['fixed'],positive_spec=family['fixed_plus'],cases=272,
            matched_frozen_nominal_safety_execution_and_rng_root=True,
            cells=dict(cells),discordant_cases=cells['10']+cells['01'],
            types_with_both_direction_preferences=[t for t,v in sorted(by_type.items())if v['10']and v['01']],
            same_type_and_nominal_strata_with_both_preferences=[list(k)for k,v in sorted(by_stratum.items())if v=={'10','01'}],
            both_success_cases=len(paired_times),mean_absolute_completion_difference_seconds=statistics.mean(differences),
            median_absolute_completion_difference_seconds=statistics.median(differences),maximum_absolute_completion_difference_seconds=max(differences),
            negative_faster_by_more_than_one_step=sum(x<-.05-1e-8 for x in paired_times),
            positive_faster_by_more_than_one_step=sum(x>.05+1e-8 for x in paired_times)),
        evidence_only=True,new_rollouts=0,training=False,global_parameter_selected=False,
        limitations=['Early-success stopping censors unexecuted candidates. Missing cells are unknown, never failure.',
            'Observed parameter sensitivity and crossed preferences do not demonstrate learnability/generalization from any particular input encoding.',
            'First-success parameter distribution is search-order dependent; it is not an optimum or an unbiased sample of all successful eta.',
            'Three free coefficients include absolute-value coupling and stateful admission; this is not a fixed linear3column matrix.',
            'The cohort informed earlier family design. This is descriptive evidence, not an independent generalization benchmark.',
            'Completion-time comparisons use only56same-case successes under both signs; first-success search does not identify the fastest successful eta.'],
        artifacts=dict(matrix=str(matrix.resolve()),matrix_sha256=sha(matrix),snapshot=str(HERE/'ETA_LEARNING_VALUE_INPUTS.json')))
    write(HERE/'ETA_LEARNING_VALUE_RESULTS.json',result)
    examples=[r for r in contrasts if r['identity']in [[4,1,'gradient_mpc',5],[4,1,'gradient_mpc',7]]]
    assert len(examples)==2 and {r['negative_success']for r in examples}=={0,1}
    result['within_same_map_and_nominal_examples']=examples
    write(HERE/'ETA_LEARNING_VALUE_RESULTS.json',result)
    text=f'''# 最佳B族是否值得学习η：已有证据复核

结论：有明确的参数选择信号，不能把各η视为近似等效。现有证据支持继续研究按状态选择η；尚未证明任何输入编码或学习器能在独立场景上泛化。本次只分析已有数据，新增rollout=0、训练=0，不恢复已删除的全局固定参数覆盖流程。

最佳已完成族为三自由系数：G、R绑定C2、T1的绝对值绑定通道放行Q（level3_01）。旧272案例是17图×8初态×2名义控制器；Oracle@64为264/272。Q有状态与输入缩放，绑定含绝对值，不能称作固定线性B的三列秩结论。

## 同一案例更换η的直接证据

- 264个成功案例中，188个在前面的η失败后才成功；成功所需首次候选序号均值{result['first_success_mean']:.3f}、中位数{result['first_success_median']:.0f}。
- 合并已有同族对照后，{len(mixed)}/272个案例具有至少一组已验证成功η和至少一组已验证失败η。
- 这证明参数会影响成败；不等于证明成功区间很窄，因为首次成功后的大部分候选未执行。

## 严格控制的旋转符号对照（仅复用旧数据）

两配置的目标系数同为0.875，分离与净空系数同为0.375，通道放行强度同为0.75，角色与放行规则也相同。唯一变化是T1从−0.75变为+0.75。272例逐一核对相同物理状态、名义控制器、随机种子、场景、时限。

| 同一案例的结果 | 案例数 |
|---|---:|
| 两个符号都成功 | {cells['11']} |
| 仅负号成功 | {cells['10']} |
| 仅正号成功 | {cells['01']} |
| 两个符号都失败 | {cells['00']} |

因此{cells['10']+cells['01']}例只改变一个系数的符号就改变成败，而且两个方向各有不可替代的已验证成功案例。共有{len(result['controlled_sign']['same_type_and_nominal_strata_with_both_preferences'])}个“场景类型×同一名义控制器”组同时包含两种相反偏好，参数选择信号不只是更换场景名称或控制器。

例如，同一张11机器人拥挤方形地图、同一个梯度下降MPC：初态5使用负号超时，正号10.60秒成功；初态7使用负号9.25秒成功，正号超时。272组对照的完整nominal/safety/execution配置与逐步PRNG根身份均已核对相同，只有修正参数改变。

两符号都成功的{len(paired_times)}例中，完成时间差的绝对值均值{statistics.mean(differences):.3f}秒、中位数{statistics.median(differences):.3f}秒、最大{max(differences):.3f}秒。这是同一案例的配对差，不是两个不同成功集合的条件均值差。

## 现有数据能回答到哪一步

已知{len(observations)}/{272*64}个“案例×候选”单元，约{100*len(observations)/(272*64):.2f}%；其余{272*64-len(observations)}个为未知。不能据此估计每个案例的成功η比例，也不能把首次成功η当成唯一正确标签或最快η。至少{len(multiple)}个案例已证实有多个成功η。

学习价值目前体现在两件事：选择正确的旋转方向、在已有正负候选中区分可解与不可解；成功后的时间也存在进一步选择空间。目标/分离/放行强度联合变化的日志存在，但不能把联合变化归因于某个单独系数。最终是否可学、能否泛化，仍需对应泛化pipeline在独立数据上验证。本次没有训练模型，也没有新增补齐候选的实验。

逐案例观测矩阵（包含observed掩码）在SSD的case_eta_observed.csv；完整初态身份和正负符号对照在controlled_sign_contrasts.json；源文件路径及哈希见ETA_LEARNING_VALUE_INPUTS.json。旧实验目录保持不变，没有从旧目录导入代码。
'''
    (HERE/'ETA_LEARNING_VALUE_REPORT.md').write_text(text)
    return result


if __name__=='__main__':
    result=run()
    print(json.dumps({k:v for k,v in result.items()if k not in ('controlled_sign','limitations','artifacts')},ensure_ascii=False,indent=2))
