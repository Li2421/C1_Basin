"""Documentation-only closeout after the source-boundary hard stop."""
from pathlib import Path
import json,hashlib,csv,datetime
H=Path(__file__).resolve().parent;R=H.parents[1];O=H.parent/'multiscene_generalization_v1'
def read(p):return json.loads(p.read_text())
def write(n,v):(H/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
NAMES=dict(A='单门隔墙',B='串联瓶颈走廊',C='多门隔墙',D='十字路口',E='环岛',F='房间与门',G='随机障碍场')

def main():
 stages=[]
 for s in ['reproduction_round40','joint_new','joint_retry_1']:
  cases=read(H/f'{s}_CASES.json');raw=[read(p) for p in (H/s/'raw').glob('*.json')];counts={k:sum(q['outcome']==k for q in raw) for k in sorted({q['outcome'] for q in raw})}
  stages.append(dict(stage=s,planned=len(cases),executed=len(raw),outcomes=counts,postflight=read(H/f'{s}_POSTFLIGHT.json'),registration=str(H/f'{s}_REGISTRATION.receipt.json')))
 write('DB_STAGE_INDEX.json',dict(executed_stages=stages,unexecuted_expert_design=dict(planned=4096,executed=0,missing=4096,preflight=str(H/'expert_gate_CACHE_PREFLIGHT.json'),postflight_status='NOT_APPLICABLE_NO_EXECUTION'),official_merger_only=True))
 catalog=read(H/'PILOT_GEOMETRY_FROZEN.json');old=read(O/'CATALOG_FROZEN.json');oldfinal=read(O/'CATALOG_FINAL.json');oldseeds={s['geometry_seed'] for x in [old,oldfinal] for ss in x['catalog'].values() for s in ss};oldcontent={json.dumps(s['rectangles']) for x in [old,oldfinal] for ss in x['catalog'].values() for s in ss}
 geochecks={}
 for f,ss in catalog.items():
  new=[s for s in ss if s['origin']=='round40 new'];training=[s for s in ss if s['role'] in ['train','train_new']];holdout=[s for s in ss if s['role']=='navigation_holdout']
  geochecks[f]=dict(training_geometries=len(training),navigation_holdout_geometries=len(holdout),reused_joint_proofs=16,new_joint_proofs=len(new),all_joint_certified=all(s['joint_feasibility']=='certified' for s in ss),new_seed_overlap_with_round39=sorted({s['geometry_seed'] for s in new}&oldseeds),new_content_overlap_with_round39=len({json.dumps(s['rectangles']) for s in new}&oldcontent),training_holdout_content_overlap=len({json.dumps(s['rectangles']) for s in training}&{json.dumps(s['rectangles']) for s in holdout}))
  assert not geochecks[f]['new_seed_overlap_with_round39'] and not geochecks[f]['new_content_overlap_with_round39'] and not geochecks[f]['training_holdout_content_overlap']
 write('GEOMETRY_PARTITION_AUDIT.json',geochecks)
 gaps=[dict(item='源码边界',reason='开工哈希审计读取了禁止的第三十轮后续两轮实际目录中66个Python文件。未写入。',evidence='SOURCE_BOUNDARY_FAILURE.json',condition='需用户明确授权从这一硬停止恢复；先修正目录排除规则，不重跑已有记录。'),dict(item='两个试点专家门',reason='各2048条仅设计登记与预检，源代码边界硬停止前未执行。成功率、碰撞率和超时率均不可估计。',condition='硬停止解除后仅执行真正缺失身份。'),dict(item='第零至三轮训练与导航门',reason='专家门未执行；没有示范数据、训练或最终检查点。',condition='专家门通过后依原顺序执行。'),dict(item='其余五族、修正实验、规模曲线与锁定测试',reason='未得到试点判定，不进入全量和修正实验。',condition='两个试点各至少两个合格实例后才进入全量；至少四个前六族达标后才进入修正实验。')]
 write('GAPS.json',gaps)
 family={f:dict(name=n,expert_gate='NOT_RUN' if f in 'BF' else 'NOT_REACHED',expert_gate_planned=2048 if f in 'BF' else 0,expert_gate_executed=0,navigation_gate='NOT_RUN',passed_instances=None,trained_instances=0,reason='source-boundary hard stop' if f in 'BF' else 'pilot verdict not available') for f,n in NAMES.items()}
 write('results.json',dict(status='HARD_STOP_SOURCE_BOUNDARY',integrity_passed=False,pilot_verdict='NOT_EVALUATED',reproduction=read(H/'REPRODUCTION_GATE.json'),geometry_partition=geochecks,families=family,expert_episodes_executed=0,training_rounds_executed=0,correction_evaluation_labels_opened=0,locked_test_evaluations=0,scaling_regret=None,gaps=gaps))
 (H/'NAVIGATION_GATE.md').write_text('本轮未执行导航能力门。串联瓶颈走廊（族 B）与房间与门（族 F）的专家门各2048条已登记和预检，均未执行；源代码边界硬停止先于示范采集与训练。通过实例数为未评估，不能写成0/4。第零轮和第三轮的检查点、失败位置图、撞墙率、超时率、路线偏离比例均不存在，不能用上一轮策略或伪造图代替。\n\n上一轮合法导航门对照：串联瓶颈走廊四个实例均0/16成功；房间与门四个实例为0、0、0、1次成功。上一轮专家候选碰撞分别257/640、134/640。新专家与39维策略没有结果，所以没有可估计的改善。旧评估几何上的本轮描述性对照也未运行。\n')
 (H/'EXPERT_GATE.md').write_text('串联瓶颈走廊（族 B）与房间与门（族 F）各64训练几何，计划每地图32回合，共各2048条；全部只完成设计与官方预检。专家门执行0条，成功、碰撞、超时、unknown计数与余量分布均未测量。没有失败位置，故不制作失败位置图。门槛仍为成功率至少99%且零撞墙，未据结果更改。\n')
 methods=['仅执行控制器','只加安全投影','全程修正参数十','全程修正参数十三','全程修正参数十五','修正参数十三至十按比例切换','修正参数十五至十按比例切换','按进度切换','通用持续右手规则','零样本选择模型','少量标签选择','完整菜单交叉拟合最优参照','专家']
 rows=[]
 for f,n in NAMES.items():rows.append([n]+['缺口：未评估']*len(methods)+['源代码边界硬停止','未产生终点证据'])
 rows.append(['前六族合并']+['缺口：未评估']*len(methods)+['试点未评估','未产生终点证据'])
 header=['子项']+methods+['原因','证据等级']
 with (H/'AXIS_TABLE.csv').open('w') as fp:
  w=csv.writer(fp);w.writerow(header);w.writerows(rows)
 (H/'AXIS_TABLE.md').write_text('| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+''.join('| '+' | '.join(row)+' |\n' for row in rows)+'\n碰撞率、超时率、未知率、平均与最大遗憾、灾难数、伤害数及置信区间均未评估，不以零代替。专家列指导航或性能参照；联合几何可解证明单独列于几何审计，不充作预算匹配的性能。\n')
 with (H/'SCALING_CURVE.csv').open('w') as fp:
  w=csv.writer(fp);w.writerow(['训练族数','平均遗憾','状态']);w.writerows([[m,'NA','缺口：试点未评估'] for m in [0,1,3,5]])
 begin=read(H/'INTEGRITY_BEGIN.json');end=read(H/'INTEGRITY_END.json');forbidden=read(H/'SOURCE_BOUNDARY_FAILURE.json')['forbidden_directories']
 before={p:v for p,v in begin['sources'].items() if not any(p.startswith(x) for x in forbidden)};after=end['sources'];parallel=set(begin['parallel_task_directories'])
 def parallel_path(p):
  try:parts=Path(p).relative_to(R/'diagnostics').parts;return bool(parts and parts[0] in parallel)
  except ValueError:return False
 changes=[dict(path=p,before=before.get(p),after=after.get(p),parallel=parallel_path(p)) for p in sorted(set(before)|set(after)) if before.get(p)!=after.get(p)]
 write('INTEGRITY_FINAL_COMPARISON.json',dict(passed=False,hard_stop='forbidden directory read in initial hash audit',changes=changes,parallel_task_directories=sorted(parallel),nonparallel_changes=[v for v in changes if not v['parallel']],MAC_equal=begin['MAC']==end['MAC'],Basin_status_equal=begin['Basin']['status']==end['Basin']['status'],Basin_diff_equal=begin['Basin']['diff_sha256']==end['Basin']['diff_sha256'],note='End audit fixes pruning; forbidden hashes retained only in original immutable start evidence; forbidden directories never reread. Changes in predeclared parallel directories are separated and do not cause failure.'))
 first='完整性审计未通过：开工审计误读了禁止的第三十轮后续两轮目录中的66个Python文件以计算哈希，没有写入这些目录；按源码边界要求已停止整轮。串联瓶颈走廊（族 B）与房间与门（族 F）的试点均未评估，导航能力门通过实例数未知，不能记为零。历史执行链16条逐位复现通过，两个试点各64个训练几何和8个新导航留出几何已具备联合可解证明；各2048条专家门仅完成登记与预检，执行0条。其余五族、规模曲线与随机障碍场锁定测试均未进入。'
 paras=[first,'本轮没有得出“试点不通过”的实验结论。停止原因是审计代码边界错误，发生在专家门之前；不能把缺失结果解释为导航能力不足。SOURCE_BOUNDARY_FAILURE.json记录了实际涉及目录与文件数量，结束审计已将这两个实际目录名加入剪枝，未再次读取。未经用户授权，不应自动恢复。','导航场、跟随点与39维观测已在本轮目录实现，静态单元测试通过，但覆盖仍有限，尚未用闭环专家门验证。固定栅格0.05米、余量目标0.15米、权重4，跟随点从双线性梯度及8邻域回退取得；专家仍为0.3米前视、原速度与目标减速，不改物理或成功标准。geometry.py、single_agent.py与原版逐字节一致，上一轮等价性测试按提示要求复用，不重做。','复现的十六条记录来自机器人数量泛化研究的双机器人确定性潜势控制器，只加安全投影：首动作字节、终止、步数、全位置轨迹全部一致。复现记录中一次成功、十五次超时；超时不称死锁。复制的runtime.py与controllers.py逐字节一致，core只增加本轮CUDA缓存路径。','两个试点各复用十六个旧训练几何及联合证据，新补四十八个训练几何和八个导航留出几何。新几何种子和矩形内容与上一轮所有角色、旧合法留出均无交集，本轮训练与留出也无内容交集。新联合参照112个候选中110个成功、2个未找到顺序路线；后两条保持unknown，取新种子补得2个成功。联合参照预算为原默认10000步，只证明几何联合可解，不充作控制器评估性能。','数据库执行链完成官方设计登记、预检、仅缺失身份、默认journal、单进程官方合并与后检；已执行130条记录，其中2条未知保留。专家门4096条仅登记与预检，未执行；不应对这批未执行计划声称后检已完成。早先联合参照作业23524因复现收据尚未存在，被worker守卫在任何物理步之前拒绝，日志全部保留；复现通过后才提交23540真正执行。','没有采集训练示范，没有训练任何第零至三轮策略，也没有本轮导航检查点。仓内复制的旧训练脚本和策略代码保留为来源参照，仍为35维，不能当成已完成的39维训练入口。专家门、导航门、旧评估几何上的本轮描述性对照、各轮失败位置图与训练成功率均缺口，见EXPERT_GATE.md、NAVIGATION_GATE.md和GAPS.json。','修正标签没有生成或打开，随机障碍场锁定评估0次；不存在修正偏好、相对只加安全投影的成功率提升、遗憾或规模趋势。AXIS_TABLE与规模曲线横坐标保留缺口，数值不填零。','资源仅使用处理器分片，图形处理器分片为零；本轮每次提交前都查队列并受十八个处理器总额度约束。队列、提交参数和中止日志保留。未安装包，未做Git写操作，现有源码没有被本轮修改。其他并行目录的变化单列；本轮完整性失败明确来自禁止目录的读取，不能用并行变化豁免。']
 (H/'REPORT.md').write_text('\n\n'.join(paras)+'\n')
 (H/'HANDOFF.md').write_text('任务：执行 docs/prompt_round40_multiscene_navigation_competence_20261017.md，先做串联瓶颈走廊（族 B）与房间与门（族 F）试点。当前状态为源码边界硬停止，不是试点实验失败。\n\n'+first+'\n\n独立接手顺序：REPORT.md、SOURCE_BOUNDARY_FAILURE.json、GAPS.json、INTEGRITY_FINAL_COMPARISON.json；再读PREREGISTRATION.md与收据、DB_STAGE_INDEX.json、PILOT_GEOMETRY_FROZEN.json与GEOMETRY_PARTITION_AUDIT.json。Python为 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python。仅本轮目录可写，官方数据库追加为唯一例外，不访问其他分布式线目录。\n\n本轮没有运行中的作业。专家计划expert_gate有4096条已登记、预检但未执行，旧未知不重跑。若用户明确授权恢复，先再次阅读最新提示与补充决定；保留已过复现与几何证据，用官方预检确认仍缺失后再执行专家门。不重做同身份rollout。两族均需达到>=99%且零撞墙后才采集示范和训练；两试点都至少两个实例通过15/16导航门才进入其余五族。当前不能直接执行复制的旧35维训练脚本，应先完成真正的39维训练、三轮专家重标注与必要测试，并在数据前冻结详细身份。\n\n开工审计遗漏的是两个实际目录名；修正后的setup_audit.py已排除。初始审计含禁止文件哈希的原证据保留，不再读取这些目录。不得把本轮静态测试通过解释为专家门通过。不得把原始十六训练几何之外的新地图联合证明解释为导航策略表现。\n')
 write('FINAL_DELIVERY.receipt.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='HARD_STOP_SOURCE_BOUNDARY',files={str(H/n):sha(H/n) for n in ['REPORT.md','HANDOFF.md','GAPS.json','results.json','SOURCE_BOUNDARY_FAILURE.json','DB_STAGE_INDEX.json','NAVIGATION_GATE.md','EXPERT_GATE.md','AXIS_TABLE.md','AXIS_TABLE.csv','SCALING_CURVE.csv','INTEGRITY_FINAL_COMPARISON.json','PILOT_GEOMETRY_FROZEN.json','GEOMETRY_PARTITION_AUDIT.json']}))
 print(first)
if __name__=='__main__':main()
