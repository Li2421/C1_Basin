from core import *
import csv,subprocess
from shared_rollout_db.src.rollout_db import connect
from shared_rollout_db.src.planner import preflight
prereg();gate();p=read(HERE/'protocol.json');cases=read(HERE/'main_CASES.json');assert len(cases)==384
write(HERE/'STAGE0_ENDPOINT_OPENING.receipt.json',dict(utc=utc(),all_required_raw_present=all((HERE/'main/raw'/f"{c['rollout_uid']}.json").exists() for c in cases),stage1_labels_not_generated=True))
names=p['controller_order'];Y=np.zeros((2,3,32,2),int);U=Y.copy();compact=[]
with connect(True) as db:
 for c in cases:
  r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json");stored=db.execute('SELECT success,numerical_failure,outcome,episode_length FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert stored and tuple(stored)==(r['success'],r['numerical_failure'],r['outcome'],r['episode_length']);assert sha(r['trace_path'])==r['trace_sha256'];i=names.index(c['base_controller_name']);j=p['arms'].index(c['arm']);Y[i,j,c['state_index'],c['seed']]=r['success'];U[i,j,c['state_index'],c['seed']]=r['numerical_failure'];compact.append({k:r[k] for k in ['case','outcome','success','collision','numerical_failure','minimum_swept_clearance','trace_path','trace_sha256']})
post=preflight(HERE/'main_PLANNED_ROLLOUTS.json'); known_unknown=int(U.sum()); assert post['summary']['genuinely_missing']==known_unknown;write(HERE/'main_POSTFLIGHT.json',dict(status='PASS' if post['summary']['genuinely_missing']==known_unknown and post['summary']['ambiguous']==0 else 'FAIL',preflight=post,known_unknown_retained=known_unknown,no_unknown_rerun=True));assert read(HERE/'main_POSTFLIGHT.json')['status']=='PASS';table=[]
for i,name in enumerate(names):
 counts=Y[i].sum((1,2));unknown=U[i].sum((1,2));best=int(np.argmax(counts));table.append(dict(controller=name,group=p['cohort'][i]['group'],success_counts=counts.tolist(),unknown_counts=unknown.tolist(),denominator=64,success_lower=(counts/64).tolist(),success_upper=((counts+unknown)/64).tolist(),best_arm=p['arms'][best]))
primary=[x for x in table if x['group']=='best'];assert len(primary)==1
result=dict(stage0_status='REJECTED',headroom_gate_passed=False,compatible_best_count=1,eligible_final_count=1,condition_a_qualifying_count=0,condition_b_qualifying_count=int(max(primary[0]['success_counts'])<=.85*64),threshold_count=3,stage1='NOT_RUN',stage2='NOT_RUN',broader_Four_conclusion='INCONCLUSIVE: compatibility gate leaves fewer than3 controllers; cannot establish absence across22 inventory runs',operational_stop_reason='Both preregistered stage0 gates cannot pass with1 compatible best (combined2 also below3)',arms=p['arms'],table=table,total_full_rollouts=len(cases),unknown=int(U.sum()),consistency_gate_cases=8)
write(HERE/'results.json',result);np.savez_compressed(HERE/'LABELS.npz',Y=Y,U=U,controller_names=np.array(names),arms=np.array(p['arms']))
with (HERE/'CENSUS.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['controller','group','arm','success','unknown','denominator','lower','upper'])
 for row in table:
  for j,arm in enumerate(p['arms']):w.writerow([row['controller'],row['group'],arm,row['success_counts'][j],row['unknown_counts'][j],64,row['success_lower'][j],row['success_upper'][j]])
sys.path.insert(0,str(R9/'vendor'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,3.5));x=np.arange(3)
for i,row in enumerate(table):ax.bar(x+(i-.5)*.3,np.array(row['success_lower'])*100,width=.3,label=row['group'])
ax.set_xticks(x,p['arms']);ax.set_ylim(0,105);ax.set_ylabel('Success lower bound (%)');ax.legend();fig.tight_layout();fig.savefig(HERE/'census.png',dpi=160);fig.savefig(HERE/'census.pdf');plt.close(fig)
# Full simulation evidence only; Four has no native deadlock detector.
from new_benchmark_common import batch_videos as official
formal=[];exported=[]
def safe(r):return r['success'] and not r['collision'] and not r['numerical_failure'] and r['minimum_swept_clearance'] is not None and r['minimum_swept_clearance']>=0
safe_rows=sorted([r for r in compact if safe(r)],key=lambda r:r['case']['rollout_uid']);used=set()
for r in safe_rows:
 c=r['case'];identity=(c['state_uid'],c['seed'])
 if identity in used:continue
 used.add(identity);role='safety_success' if not exported else f'diagnostic_success_{len(exported)}';meta,data=official.load_trace(Path(r['trace_path']));path=HERE/'videos'/f'{role}.mp4';frames=official.render_video(path,meta,data);exported.append(dict(role=role,formal_role=role=='safety_success',file=str(path),sha256=sha(path),frames=frames,trace=r['trace_path'],trace_sha256=r['trace_sha256'],provenance=meta))
 if role=='safety_success':formal.append(dict(role=role,trace=r['trace_path']))
 if len(exported)==3:break
write(HERE/'video_plan.json',dict(batch_id=EXP,scenarios={'four_way_intersection':formal}));official.SCENARIOS=('four_way_intersection',)
try:official.main(['--plan',str(HERE/'video_plan.json'),'--output',str(HERE/'videos/strict_gate')]);rc=0;error=None
except Exception as ex:rc=2;error=str(ex)
missing=[x for x in official.ROLES if x not in {r['role'] for r in formal}];write(HERE/'videos/manifest.json',dict(batch_id=EXP,complete=rc==0,videos=exported,missing_roles=missing,official_exit_code=rc,error=error));write(HERE/'VIDEO_GATE.json',dict(utc=utc(),passed=rc==0,exit_code=rc,missing_roles=missing,scope='Four-only isolated wrapper; original roles/load/render/validator unchanged',source_sha256=sha(ROOT/'new_benchmark_common/batch_videos.py')))
rows='\n'.join(f"| {row['group']} | "+' | '.join(f"{n}/64 ({100*n/64:.4f}%)" for n in row['success_counts'])+' |' for row in table)
text=f'''# Round14B Four controller census\n\nFour 的跨 controller η 泛化问题在本轮兼容队列中**未建立**：阶段0两项 headroom 门均不通过，按预注册停止。仅1个 best 与1个参数不同的 final 可用，无法推断全部 Four controller 都不存在泛化问题；阶段压缩与 E 可学习性均 **INCONCLUSIVE / 未执行**。\n\n实际磁盘找到22套 best/final，共44文件，非提示所述23套。严格比较 E8 v13 broad 的网络参数形状、输入框架、环境 fingerprint 及四项归一化数组后，仅该 run 的 best/final 一致，其余21个best不可用（详情 COMPATIBILITY_METADATA_AUDIT.json）；未替换或重拟合归一化器。两者参数不同，final单独为次要组。前向同key重复计算字节一致，导入后显式恢复FP32；历史8例终止原因、步数、第一动作字节完全复现。\n\n32个全新真实t0 state，F10/F13/F15，各seed0/1；384条完整rollout，unknown {int(U.sum())}。旧DB及标签按相同语义保留，标准preflight384missing，单进程merger后全部请求有记录；标准planner把unknown列为missing，本轮保留且不重跑；门8例另计。State去重审计记录不可恢复的历史证据，不声称这些证据绝对无重叠。\n\n| group | F10 | F13 | F15 |\n|---|---|---|---|\n{rows}\n\n阶段0(a)需要至少3个非多数regime且差>=10pp的controller，(b)需要至少3个最优固定臂<=85%的controller。主组只有1个，合计两组也只有2个，不可能达到任何人数门。阶段0判定 **REJECTED**（进入阶段1的假设）；广义Four跨controller问题 **INCONCLUSIVE**（兼容性导致证据不足）。未放宽门槛，未新生成探测或时间表标签。\n\n交付3个完整仿真视频，其中1个正式safety_success与2个诊断成功案例。Four没有原生deadlock判据，deadlock_show / deadlock_resolution 缺失；统一入口已执行，退出码2，视频交付**未完成**，无COMPLETE标记。timeout不称为deadlock。\n\n所有新产物限本目录，共享DB仅标准生命周期写入。历史文件未改动；无新训练。资源与Round14A合计不超过24CPU/3GPU。\n'''
(HERE/'REPORT.md').write_text(text);(HERE/'HANDOFF.md').write_text(text+'\n下一步若要覆盖其它21个best，必须先明确是否授权 checkpoint 自带归一化器而非与E8数值相同；当前提示下不得强行运行。已有标签不因表示改变而失效。\n')
write(HERE/'DELIVERY.receipt.json',dict(utc=utc(),numerical_status='STAGE0_STOP',video_status='INCOMPLETE',files={str(q.relative_to(HERE)):sha(q) for q in [HERE/'REPORT.md',HERE/'HANDOFF.md',HERE/'results.json',HERE/'CENSUS.csv',HERE/'videos/manifest.json']},new_full_rollouts=384,gate_rollouts=8))
print(json.dumps(result));sys.exit(rc)
