"""Record an interface-blocked run without inventing experimental evidence."""
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
begin = json.loads((HERE / 'INTEGRITY_BEGIN.json').read_text())
end = json.loads((HERE / 'INTEGRITY_END.json').read_text())
changes = {k: {'begin':begin[k], 'end':end[k]} for k in ['Basin','MAC'] if begin[k] != end[k]}
source_changes = {k: [begin['sources'].get(k), end['sources'].get(k)]
                  for k in begin['sources'].keys() | end['sources'].keys()
                  if begin['sources'].get(k) != end['sources'].get(k)}
passed = not changes and not source_changes
write('INTEGRITY_COMPARISON.json', dict(passed=passed, repository_changes=changes,
    source_changes=source_changes, forbidden_directories_not_traversed=True,
    imported_existing_source_coverage='All existing Python source files in allowed trees covered at start'))
families = ['单门隔墙（族 A）','串联瓶颈走廊（族 B）','多门隔墙（族 C）',
            '十字路口（族 D）','环岛（族 E）','房间与门（族 F）','随机障碍场（锁定测试族 G）']
reason = '真正单机器人 episode 与只重写初始化、继承原 step 的限制存在接口冲突，等待用户明确最小例外。'
gaps = [dict(family=f, capability='NOT_RUN', reason=reason) for f in families]
gaps += [dict(item='历史执行十六条逐位复现', reason='尚未执行；开工与后续队列检查均见其他任务占满十八个 CPU shard，本轮没有提交作业。'),
         dict(item='几何清单、等价性测试、训练与评估', reason='仍在阶段一接口审查，尚未进入；不把未运行的等价性测试记为失败。'),
         dict(item='数据库、预注册、模型、规模曲线', reason='没有登记新身份或产生新 rollout；完整预注册、逐条历史标签兼容性与去重、官方缓存流程尚未开展。')]
write('GAPS.json', gaps)
write('results.json', dict(status='BLOCKED_INTERFACE_AWAITING_USER', completed=False,
    integrity_passed=passed, endpoints={str(k):'NOT_EVALUATED' for k in range(1,5)},
    families=gaps[:7], scaling_curve=[dict(training_families=m,regret_pp=None) for m in [0,1,3,5]],
    locked_test=dict(opened=False,evaluations=0,result=None),
    database=dict(new_registrations=0,new_rollouts=0,new_journals=0),
    equivalence_test='NOT_RUN', source_exception_implemented=False))
audit = '完整性审计通过（允许读取的既有源码、仓库状态与差异哈希起终一致；禁止目录未遍历）' if passed else '严格完整性审计未通过（差异见 INTEGRITY_COMPARISON.json）'
paragraph = audit + '。本轮在单机器人接口冲突处受阻，尚未完成实验：多场景终点一至四均未评估，无成功率或遗憾数值；单门隔墙、串联瓶颈走廊、多门隔墙、十字路口、环岛、房间与门及随机障碍场的导航能力门和多机器人能力门均未运行，修正作用与动作偏好未知；训练族数为零、一、三、五时的规模曲线尚无结果；随机障碍场锁定测试未打开、评估零次。'
detail = '''

这是受阻状态交付，不是实验完成报告，也没有确认性结论。所有新文件仅在本轮目录。没有 Git 写操作、包安装或共享数据库写入；没有读取禁止的并行研究目录。队列快照保存在完整性审计文件中；本轮未提交任何作业。

阻塞来自 prompt 第二节的“只重写 __init__，step 全部继承”与第三节的“场景里只有一个机器人”。原 Config 在 scenario.py 第三十八行拒绝单机器人，原 BottleneckEnv.step 在 environment.py 第八十二行对空机器人对求最小值。异常证据见 INTERFACE_CHECK.json，具体最小处理方案见 INTERFACE_CONFLICT.md。没有修改原文件，也没有擅自实现例外。

生成环境与原环境的等价性测试尚未运行，不能写成等价性测试失败。历史报告中的复现通过也未冒充本轮复现。历史检索与兼容性边界见 HISTORY_AUDIT.md。全部缺口见 GAPS.json；统计表中的缺口不是零成功率、不是 unknown 标签，也不是 timeout 或 deadlock。死锁修复仍应按成功率相对只加安全投影的提升衡量。

已向用户请求明确：是否允许本轮目录内单独的单机器人环境，仅定义单机器人配置及空碰撞对的无穷距离；四机器人生成环境仍完全继承原 step。答复前保留全部目标标签关闭。收到允许后从阶段一继续，不跳过历史执行复现、生成环境等价性测试、完整预注册或任何数据库检查。
'''
(HERE/'REPORT.md').write_text(paragraph + detail)
(HERE/'HANDOFF.md').write_text('当前任务是执行 docs/prompt_round39_multiscene_generator_cross_scene_20261017.md；状态为接口受阻，尚未完成。\n\n'+paragraph+detail)
methods = ['只执行控制器动作','只加安全投影','全程修正参数十','全程修正参数十三','全程修正参数十五',
           '修正参数十三至十按比例切换','修正参数十五至十按比例切换','修正参数十三至十按进度切换','通用持续右手规则',
           '场景无关零样本选择模型','源场景平均最佳臂','少量标签经验选择','少量标签安全选择',
           '四个先导状态简化选择','最佳固定参数交叉拟合参照','完整菜单交叉拟合参照','样本内完整菜单参照',
           '原冻结控制器级选择模型','原冻结不看控制器的选择模型','探测估计规则选择','顺序通行专家']
extra = ['碰撞率','超时率','平均遗憾','最大遗憾','灾难数','伤害数','证据等级','原因']
fields = ['目标族'] + methods + extra
rows = []
for name in families[:6] + ['前六族合并'] + families[6:]:
    row = {k:'缺口：未运行' for k in fields}
    row.update({'目标族':name,'证据等级':'无新实验结论','原因':reason})
    rows.append(row)
with (HERE/'AXIS_TABLE.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
(HERE/'AXIS_TABLE.md').write_text('| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n'+''.join('| '+' | '.join(row[k] for k in fields)+' |\n' for row in rows))
write('STATUS.receipt.json', dict(utc=datetime.now(timezone.utc).isoformat(),completed=False,
    files={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
           ['REPORT.md','HANDOFF.md','results.json','GAPS.json','AXIS_TABLE.csv','INTERFACE_CHECK.json']},
    user_exception_response_received=False))
print(paragraph)
