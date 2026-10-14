from core import *
from collections import Counter
prereg();gate();r=read(HERE/'results.json');trans=read(HERE/'FAILURE_TRANSITIONS.json');rows=[]
for sc in SCENES:
    for baseline,schedule in [('F10','S13_t2'),('F13','S13_t2'),('F10','S15_t2'),('F15','S15_t2')]:
        counts=Counter()
        for q in trans:
            if q['scene']==sc and q['baseline']==baseline and q['schedule']==schedule:counts.update(q['counts'])
        assert sum(counts.values())==480;rows.append(dict(scene=sc,baseline=baseline,schedule=schedule,counts=dict(counts),timeout_to_success=counts['timeout -> success'],deadlock_to_success=counts['deadlock -> success'],success_to_timeout=counts['success -> timeout'],success_to_deadlock=counts['success -> deadlock'],unknown_in_pair=sum(v for k,v in counts.items() if 'numerical_failure' in k)))
write(HERE/'PAIRED_FAILURE_CHANGE_SUMMARY.json',rows)
header='| 场景 | 对照 | timeout→success | deadlock→success | success→timeout | success→deadlock | 含unknown的pair |\n| --- | --- | --- | --- | --- | --- | --- |\n'
body='\n'.join('| '+' | '.join(map(str,[q['scene'],q['baseline']+'→'+q['schedule'],q['timeout_to_success'],q['deadlock_to_success'],q['success_to_timeout'],q['success_to_deadlock'],q['unknown_in_pair']]))+' |' for q in rows)
text='''\n## 逐场景失效改变与视频缺口复核\n\n下表为X3预注册的t2配对比较，每行480个同state/controller/seed pair。timeout保持原事件名，只有原判据触发才记deadlock；含unknown的pair不当作已知失败转换。\n\n'''+header+body+'\n\n'
text+='Toy最优已登记时间表S13@t1只比最佳固定点率高1.250pp，t2反而明显受损；DB的S15@t2接近F15（低1.042pp），S13@t2显著受损；Four没有超过最佳固定的时间表点率，固定F13却极差；Ring最佳S15@t1较最佳固定高9.792pp，但LOSO源仍选F15而在Ring失败。这些是本批次描述性事实，不是事后调整动作或新的确认组。双阶段均≥10pp数量Toy0/6、DB1/6、Four0/6、Ring1/6，未满足“各场景都有阶段结构”的描述条件；也未满足“Toy/DB/Four所有时间表均不优于固定”的条件（Toy有小幅点增益）。提示的结论映射并非穷尽所有结果，本轮不支持通用同源解释，不能据此断言唯一替代根因或H_compress仅在Ring存在。\n\n'
audit=read(HERE/'VIDEO_CANDIDATE_AUDIT.json');db=audit['scene_audits']['double_bottleneck'];text+=f"视频全量候选复核：Toy有{audit['scene_audits']['toy_give_way']['distinct_baseline_UIDs']}个可配对的不同真实死锁baseline，DB只有{db['distinct_baseline_UIDs']}个baseline（{db['qualified_pairs']}个成功时间表干预共享同一baseline和初态seed）；不把同一死锁基线的两个干预充作两个不同死锁案例。Four/Ring没有真实deadlock候选。12个完整视频中正式角色7个、timeout诊断5个；DB缺deadlock_resolution，Four/Ring各缺两个deadlock角色，统一入口非零，视频交付未完整。\n\n"
compare=read(HERE/'INTEGER_REPAIR_RESULT_COMPARISON.json');text+='整数边界修复后，真实数据X1/X2主状态、字面点门和全部LOSO源选择与保留首版一致，见INTEGER_REPAIR_RESULT_COMPARISON.json。原标签、raw/trace不变。\n'
for name in ['REPORT.md','HANDOFF.md']:
    path=HERE/name;path.write_text(path.read_text()+text)
old=read(HERE/'DELIVERY.receipt.json');freeze(HERE/'DELIVERY_INITIAL.receipt.json',old);old.update(utc=utc(),report_sha256=sha(HERE/'REPORT.md'),handoff_sha256=sha(HERE/'HANDOFF.md'),final_details_added=True,initial_delivery_receipt_sha256=sha(HERE/'DELIVERY_INITIAL.receipt.json'));write(HERE/'DELIVERY.receipt.json',old)
print('Paired failure descriptions and formal-video gap audit added; statistical results unchanged')
