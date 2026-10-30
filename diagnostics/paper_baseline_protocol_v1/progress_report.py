"""Small phase receipts; no outcome-driven control or selection."""
import guard
from guard import HERE
import json,datetime


def update():
    names=['NATIVE_REPRODUCTION_RESULTS.json','PORT_FIX_BEFORE_AFTER.json','NARROW_PASSAGE_DIAGNOSTIC.json',
           'CONSISTENT_BOX_SCENES.json','PORT_GATES.json','ENV_REUSE_PHASE_COMPLETE.json',
           'PARAMETERS_FROZEN.json','FORMAL_TEST_STARTED.json','CONSISTENT_OURS_RESULTS.json']
    phases={}
    for name in names:
        path=HERE/name
        if path.exists():
            value=json.loads(path.read_text());phases[name]=dict(present=True,complete=value.get('complete'),
                timestamp=value.get('time'),modified_ns=path.stat().st_mtime_ns)
        else:phases[name]=dict(present=False)
    budget=json.loads((HERE/'BUDGET.json').read_text())if(HERE/'BUDGET.json').exists()else{}
    tuning=json.loads((HERE/'TUNING_PROGRESS.json').read_text()).get('results',{})if(HERE/'TUNING_PROGRESS.json').exists()else{}
    selections={key:dict(complete=row.get('complete'),available=row.get('available'),index=row.get('global_index'),
        expanded=row.get('saturation',{}).get('expand'),selected_training_score=row.get('final_ranking',[None])[0])for key,row in tuning.items()}
    live=json.loads((HERE/'LIVE_STATUS.json').read_text())if(HERE/'LIVE_STATUS.json').exists()else{}
    value=dict(time=datetime.datetime.now().astimezone().isoformat(),phases=phases,budget=budget,
               training_selections=selections,completed_training_methods=sum(row.get('complete')is True for row in selections.values()),
               planned_training_methods=6,live=live,
               final_report_complete=(HERE/'COMPLETE.json').is_file(),no_static_artifact_claimed_as_rollout_result=True)
    (HERE/'PROGRESS_RESULTS.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    if not(HERE/'SUMMARY_TABLES.json').exists():
        (HERE/'RESULTS.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        native=json.loads((HERE/'NATIVE_REPRODUCTION_RESULTS.json').read_text())
        lines=['# 论文基线轮进展','',
            '原环境复现保留死锁感知控制和旋转木马；Chen因默认场景帧间连续碰撞退出，其余三个无作者代码/参数的方法退出。方形与十字验证/测试的物理一致初态合法率均128/128，主表按预登记规则使用一致版本；冻结学习模型训练于旧版本，不重训。','',
            '正式调参和测试以冻结文件及下表为准；不把单机诊断当作正式成功率。','',
            '| 阶段文件 | 已产生 | 完成标记 |','|---|---|---|']
        for name,row in phases.items():lines.append(f"| {name} | {row['present']} | {row.get('complete','—')} |")
        if selections:
            lines+=['','## 训练集选参进度（非独立测试成绩）','', '| 基准与方法 | 完成 | 选中组 | 扩到32组 | 选中组训练成功率 |','|---|---|---:|---|---:|']
            for key,row in selections.items():
                score=row.get('selected_training_score')or{}
                from method_names import NAMES
                benchmark,method=key.split('::')
                name=('基准甲'if benchmark=='A'else'基准乙')+'：'+NAMES.get(method,method)
                lines.append(f"| {name} | {row['complete']} | {row['index']} | {row['expanded']} | {score.get('scene_equal_success','缺失')} |")
            lines+=['',f"当前批次：{live.get('stage','未知')}，{live.get('stage_complete','—')}/{live.get('stage_total','—')}；CPU实际{live.get('active_user_cpus','—')}、上限{live.get('cpu_limit','—')}。"]
        path=HERE/'PORT_FIX_BEFORE_AFTER.json'
        if path.exists():
            data=json.loads(path.read_text());lines+=['','## 返回值修复训练单机对照','', '| 方法 | 基准 | 修复前成功/总数 | 修复后成功/总数 | 前/后未处理异常 |','|---|---|---:|---:|---:|']
            from method_names import NAMES
            for group in data['groups']:
                def total(label,key):return sum(row[label][key]for row in group['categories'])
                lines.append(f"| {NAMES[group['method']]} | {group['benchmark']} | {total('before','success')}/{total('before','planned')} | {total('after','success')}/{total('after','planned')} | {total('before','unhandled')}/{total('after','unhandled')} |")
        lines+=['',f"新增尝试预留{budget.get('actual_attempts',0)}/50000；实际轨迹、逐候选计划、preflight与官方journal保留。",'',
                '原环境离散/连续安全差异、依赖修复、参数出处与当前缺口详见NATIVE_REPRODUCTION.md、MARGIN_ALIGNMENT.md、GAPS.md。']
        (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return value


if __name__=='__main__':update()
