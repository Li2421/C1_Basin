"""Deliver actual two-family expert-gate failures, after official postflight."""
from common import *
import csv
NAMES=dict(A='单门隔墙',B='串联瓶颈走廊',C='多门隔墙',D='十字路口',E='环岛',F='房间与门',G='随机障碍场')

def protected_status(status,peers):
 keep=[]
 for line in status.splitlines():
  path=line[3:].split(' -> ')[-1].strip('"');parts=Path(path).parts
  if len(parts)>=2 and parts[0]=='diagnostics':
   name=parts[1]
   if name==HERE.name or name in peers or name.startswith('teacher_student_') or (name.startswith('decentralized_') and name!='decentralized_pipeline_v1'):continue
  keep.append(line)
 return sorted(keep)

def main():
 result=read(HERE/'EXPERT_GATE_RESULTS.json');families=result['families'];assert all(not r['passed'] for r in families.values())
 begin=read(HERE/'INTEGRITY_RESUME_BEGIN.json');end=read(HERE/'INTEGRITY_RESUME_END.json');before=begin['sources'];after=end['sources'];changes=[dict(path=p,before=before.get(p),after=after.get(p)) for p in sorted(set(before)|set(after)) if before.get(p)!=after.get(p)]
 peers=set(begin['parallel_metadata_only'])|set(end['parallel_metadata_only'])
 def is_peer(path):
  try:return Path(path).relative_to(D).parts[0] in peers
  except ValueError:return False
 protected=[v for v in changes if not is_peer(v['path'])]
 status_begin=protected_status(begin['Basin']['status'],peers);status_end=protected_status(end['Basin']['status'],peers)
 audit_passed=not protected and status_begin==status_end and begin['Basin']['head']==end['Basin']['head'] and begin['MAC']==end['MAC'] and begin['Basin']['diff_sha256']==end['Basin']['diff_sha256']
 comparison=dict(passed=audit_passed,baseline='INTEGRITY_RESUME_BEGIN.json',end='INTEGRITY_RESUME_END.json',protected_source_changes=protected,parallel_source_changes=[v for v in changes if is_peer(v['path'])],parallel_metadata_begin=begin['parallel_metadata_only'],parallel_metadata_end=end['parallel_metadata_only'],prohibited_metadata_begin=begin['prohibited_metadata_only'],prohibited_metadata_end=end['prohibited_metadata_only'],MAC_equal=begin['MAC']==end['MAC'],Basin_status_equal=begin['Basin']['status']==end['Basin']['status'],Basin_diff_equal=begin['Basin']['diff_sha256']==end['Basin']['diff_sha256'],protected_status_begin=status_begin,protected_status_end=status_end,Basin_HEAD_equal=begin['Basin']['head']==end['Basin']['head'],prior_boundary_deviation='Retained, explicitly authorized recovery; forbidden content unused and no writes; not current audit failure')
 write(HERE/'INTEGRITY_FINAL_COMPARISON.json',comparison)
 summary=[]
 for f in ['B','F']:
  r=families[f];c=r['counts'];summary.append(f"{NAMES[f]}（族 {f}）：专家成功{c['success']}/2048（{100*r['success_rate']:.2f}%），撞墙{c['collision']}，超时{c['timeout']}，未知{c['unknown']}，专家门未通过")
 first=('恢复后完整性审计通过。' if audit_passed else '恢复后完整性审计未通过，差异见审计记录。')+'试点判定不通过，按要求停止扩展。'+'；'.join(summary)+'。两个试点均无合格导航实例：未采集示范、未训练策略，导航能力门未评估。其余五族、修正实验、规模曲线与随机障碍场锁定测试均未进入。'
 gaps=[dict(item='串联瓶颈走廊导航策略',reason='固定专家未通过至少99%成功且零撞墙的前置门；不能采集示范或训练。',conditions='需另行授权新的预注册专家设计；本轮不改设置重跑。'),dict(item='房间与门导航策略',reason='固定专家未通过至少99%成功且零撞墙的前置门；不能采集示范或训练。',conditions='需另行授权新的预注册专家设计；本轮不改设置重跑。'),dict(item='第零轮与第三轮策略诊断',reason='没有进入训练，检查点、训练成功率和失败位置图不存在；不能补画为零失败。',conditions='先获得通过专家门的示范来源。'),dict(item='其余五族与多场景修正选择',reason='两个试点未通过；不进入全量、修正能力门或标签评估。',conditions='新的预注册试点两族各至少两个实例通过导航门后再扩展；至少四个前六族达标才进入修正实验。')]
 write(HERE/'GAPS.json',gaps)
 fs={f:dict(name=n,expert_gate=families[f] if f in families else None,trained_instances=0,qualified_navigation_instances=0,navigation_gate='NOT_RUN',reason='expert gate failed' if f in families else 'pilot failed; full stage not entered') for f,n in NAMES.items()}
 write(HERE/'results.json',dict(status='COMPLETE_WITH_FAILED_PILOT',integrity_passed=audit_passed,pilot_passed=False,families=fs,expert_gate=result,training_rounds_executed=0,correction_evaluation_labels_opened=0,locked_test_evaluations=0,scaling_regret=None,gaps=gaps))
 p=['专家门按冻结规则各执行64训练几何×32回合；所有原始轨迹、墙体扫掠最小余量和未知记录保留。成功率门槛为至少99%，并要求零撞墙。导航场参数、跟随点参数、物理与成功标准没有根据门结果改变。']
 for f in ['B','F']:
  r=families[f];q=r['minimum_wall_clearance_quantiles'];p.append(f"{NAMES[f]}：最低墙体扫掠余量分位数（米）为{json.dumps(q,ensure_ascii=False)}。相对原几何路径偏离超过0.5米的回合比例{r['route_deviation_episode_rate']:.4%}、状态比例{r['route_deviation_state_rate']:.4%}；该路线偏离仅是描述，因为避墙路线允许比原擦墙路径绕远。失败终点图见expert_diagnostics/{f}_expert_failure_positions.svg，颜色区分撞墙、超时与未知。")
 (HERE/'EXPERT_GATE.md').write_text('\n\n'.join(summary+p)+'\n')
 nav=['两个试点均在专家门退出，未进入第零至三轮训练或导航能力门。没有四个已训练实例可逐个评估，不能写成四个实例均失败。', '上一轮对照：串联瓶颈走廊专家碰撞257/640（40.16%），四个合法留出策略成功均0/16；房间与门专家碰撞134/640（20.94%），最佳策略1/16（6.25%）。本轮专家使用更多几何与不同任务，碰撞率对照为描述性，不能视作配对改善估计。']
 for f in ['B','F']:
  r=families[f];nav.append(f"{NAMES[f]}：本轮专家成功率{r['success_rate']:.4%}、撞墙率{r['collision_rate']:.4%}、超时率{r['timeout_rate']:.4%}、未知率{r['unknown_rate']:.4%}。第零轮策略、第三轮策略、导航留出几何结果与旧评估几何上的本轮描述性对照全部未运行，失败位置图不可提供，原因是未达到训练前置门。")
 (HERE/'NAVIGATION_GATE.md').write_text('\n\n'.join(nav)+'\n')
 methods=['仅执行控制器','只加安全投影','全程修正参数十','全程修正参数十三','全程修正参数十五','修正参数十三至十按比例切换','修正参数十五至十按比例切换','按进度切换','通用持续右手规则','零样本选择模型','第十六轮冻结的控制器级选择模型（A3）','第十六轮冻结的不看控制器的评价模型（A0）','探测估计后规则选择（G_E）','最佳固定参数交叉拟合参照','完整菜单样本内最佳参照','源训练场景平均最佳臂','少量标签选择','完整菜单交叉拟合最优参照','专家']
 header=['子项']+methods+['原因','证据等级'];rows=[]
 for f,n in NAMES.items():rows.append([n]+['缺口：未评估']*len(methods)+['专家门失败' if f in families else '试点未通过，未进入全量','未产生修正终点证据'])
 rows.append(['前六族合并']+['缺口：未评估']*len(methods)+['试点未通过','未产生修正终点证据'])
 with (HERE/'AXIS_TABLE.csv').open('w') as out:
  w=csv.writer(out);w.writerow(header);w.writerows(rows)
 (HERE/'AXIS_TABLE.md').write_text('| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+''.join('| '+' | '.join(row)+' |\n' for row in rows)+'\n专家单机器人门单独报告，不充作多机器人性能。碰撞率、超时率、遗憾、灾难数、伤害数和置信区间均未估计。\n')
 with (HERE/'SCALING_CURVE.csv').open('w') as out:
  w=csv.writer(out);w.writerow(['训练族数','平均遗憾','状态']);w.writerows([[m,'NA','缺口：试点未通过'] for m in [0,1,3,5]])
 write(HERE/'TRAINING_STAGE_STATUS.json',dict(families={f:{str(r):dict(status='NOT_RUN',new_training_transitions=0,checkpoint=None,training_success_rate=None) for r in range(4)} for f in NAMES},training_pipeline_implemented=True,training_pipeline_receipt='TRAINING_PIPELINE39.receipt.json'))
 stages=[]
 for stage in ['reproduction_round40','joint_new','joint_retry_1','expert_gate_v3']:
  raw=[read(x) for x in (HERE/stage/'raw').glob('*.json')];stages.append(dict(stage=stage,executed=len(raw),outcomes={k:sum(r['outcome']==k for r in raw) for k in sorted({r['outcome'] for r in raw})},postflight=read(HERE/f'{stage}_POSTFLIGHT.json')))
 write(HERE/'DB_STAGE_INDEX.json',dict(executed_stages=stages,old_expert_gate_design='4096 original source profiles retained, unexecuted; broadcast-only fix requires appended exact controller identities',old_unknown_preserved=True,official_workflow=True))
 paragraphs=[first,'原源码边界停止按用户补充决定恢复：旧提示词缺少实际目录名属于提示词错误。SOURCE_BOUNDARY_FAILURE.json与旧INTEGRITY_BEGIN.json原样保留。恢复后按分布式与师生线目录名前缀剪枝，协议指定的分布式流水线目录为只读例外；禁止和并行目录只记录路径、大小、修改时间，不打开内容、不计算哈希。结束审计与新的INTEGRITY_RESUME_BEGIN.json比较，既往只读边界偏差不冒充实验失败。', '十六条历史执行链复现、原几何划分及联合证据全部复用，没有重做。两个试点各64训练几何、8个新导航留出几何。导航场单元测试通过，目标、门口、墙角和批量一致性已覆盖；静态路径跟随对照分别比原几何路径长约3.6%和2.6%。这些静态检查不代表0.3米跟随点专家的闭环门通过。', '采集示范之前已实现39维策略训练入口、两类网络、固定开发集检查点选择、第三轮固定为最终策略、三轮专家重标注、训练几何污染检查和撞墙前状态标签测试；配置与身份公式已冻结，见TRAINING_PIPELINE39_V7.receipt.json。单元测试发现并修正了回退分支数组广播错误，仅增加显式广播，没有改参数或物理；随后排除回退候选中的中心格点，严格遵守八邻域定义。两处修正在任何专家回合之前完成；原未执行登记均保留，修正版追加精确控制器身份，见PREREGISTRATION_BROADCAST_FIX.receipt.json。', '专家结果与余量分布见EXPERT_GATE.md、EXPERT_GATE_RESULTS.json和EXPERT_EPISODES.csv；失败位置图读取实际轨迹生成。超时不称死锁。路线偏离比例相对原Geometry.path计算，避墙路线本来可以绕远，因此该比例不是额外失败标准，也没有用于调参。', '两个专家门均未通过，依阶段规则没有采集示范、没有运行第零至三轮训练或导航门。第零轮和第三轮的失败图及训练成功率标为未运行，不伪造空图。其余五族没有进入全量。单机器人专家失败不能作为多机器人死锁或修正方法有效性的结论。', '共享数据库只经官方设计追加登记、预检、真正缺失运行、默认journal、单进程merger与后检。所有已执行阶段见DB_STAGE_INDEX.json；未知保留，不重跑。资源不足时等待，提交前逐次检查全部用户作业，最初按十八个处理器总请求等待；用户于2026-10-18授权使用节点剩余六个后，本轮最多六个处理器分片，节点总量二十四，逐次检查全部用户队列与空闲CPU；图形处理器分片本轮使用零。没有安装包，没有Git写操作，没有修改现有受保护源码。', '多机器人能力门、探测、选择模型、少量标签、修正评估与随机障碍场锁定测试均未进入。不存在相对只加安全投影的成功率提升或时间表与持续右手规则偏好。训练族数0、1、3、5的遗憾均为缺口，不画零曲线。']
 (HERE/'REPORT.md').write_text('\n\n'.join(paragraphs)+'\n')
 (HERE/'HANDOFF.md').write_text(first+'\n\n任务为执行docs/prompt_round40_multiscene_navigation_competence_20261017.md及第零节恢复决定。只写diagnostics/multiscene_navigation_competence_v1/，共享数据库官方追加为唯一例外；禁止目录按前缀剪枝，仅协议分布式流水线目录允许只读。Python为/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python。\n\n独立接手顺序：REPORT.md、EXPERT_GATE.md、EXPERT_GATE_RESULTS.json、GAPS.json、INTEGRITY_FINAL_COMPARISON.json，然后PREREGISTRATION.md与恢复、广播修订收据，TRAINING_PIPELINE39_V7.receipt.json，DB_STAGE_INDEX.json。原停止交付在previous_source_boundary_stop_delivery/，已被用户明确授权恢复；旧失败记录与旧审计保留。当前最终状态是专家试点不通过，按用户要求停止扩展，不自动改参数重跑。\n\n本轮已实现39维训练与专家重标注代码，但没有执行训练；旧35维脚本仅作来源参照，不要直接运行。专家门实际阶段expert_gate_v3，完整4096回合及默认journal已官方合并后检。原expert_gate只登记未执行，广播修复导致精确源码身份变化，旧记录保留；不要运行它冒充新证据。已过历史复现与联合几何证据不重做。unknown保留，超时不称死锁。\n\n若未来另行研究新的导航专家，必须先获新研究授权并新预注册；本轮不按结果调整设置。不应把静态导航场函数通过或联合可解证明当成闭环专家门通过。规模曲线与锁定测试缺口没有被检验，不能宣称泛化有效或无效。\n')
 write(HERE/'ACTIVE_STAGE.json',dict(utc=utc(),status='COMPLETE_WITH_FAILED_PILOT',pilot_passed=False))
 names=['REPORT.md','HANDOFF.md','results.json','GAPS.json','EXPERT_GATE.md','EXPERT_GATE_RESULTS.json','EXPERT_EPISODES.csv','NAVIGATION_GATE.md','AXIS_TABLE.md','AXIS_TABLE.csv','SCALING_CURVE.csv','TRAINING_STAGE_STATUS.json','DB_STAGE_INDEX.json','INTEGRITY_FINAL_COMPARISON.json']
 write(HERE/'FINAL_DELIVERY.receipt.json',dict(utc=utc(),status='COMPLETE_WITH_FAILED_PILOT',files={str(HERE/n):sha(HERE/n) for n in names}))
 print(first)
if __name__=='__main__':main()
