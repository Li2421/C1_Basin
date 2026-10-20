"""Complete publication prose/tables from frozen predictions; no new simulation."""
from core import *
from guard import guard
from audit_utils import source_inventory
from analysis import FULL
import shutil
G=guard()
LABEL={'nominal only':'只跑学生、不加安全层与修正（nominal only）','safety baseline':'只加安全层、修正系数为零（safety baseline）','F10':'全程使用第10组修正参数（F10）','F13':'全程使用第13组修正参数（F13）','S300':'前300步使用第13组、之后第10组修正参数的时间表（S300）','S500':'前500步使用第13组、之后第10组修正参数的时间表（S500）','RHR-fair':'公平调参右手规则：无进展触发、2秒、0.25米每秒（RHR-fair）'}
def main():
 assert (HERE/'DELIVERY.receipt.json').exists();r=read(HERE/'results.json');z=np.load(HERE/'analysis_arrays.npz');arms=z['arms'].tolist();names=z['controller_names'].tolist();dest=HERE/'automated_delivery';dest.mkdir(exist_ok=False)
 for n in ['REPORT.md','HANDOFF.md','results.json','PAPER_TABLE.md','FAILURE_COMPOSITION.md']:shutil.copy2(HERE/n,dest/n)
 for v in ['V0','V1','V3']:
  ids=r['eligible'][v]
  if not ids:continue
  s=z[v+'_success'][ids];col=z[v+'_collision'][ids];tim=z[v+'_timeout'][ids]
  masks={'自身停滞':(tim[...,0]==1)&(col[...,0]==0),'安全层导致的停滞':(col[...,0]==1)&(tim[...,1]==1)}
  for kind,mask in masks.items():r['failure_composition'][v][kind]['rescued']={a:int((mask&(s[...,j]==1)).sum()) for j,a in enumerate(arms)}
 lines=['各列使用通过两个门且完成主评估的对应学生组；组间数量不同，主检验另用同教师配对交集。成功率及95%置信区间为unknown下界，完整上界见results.json；碰撞、超时均为已知事件占全部局的比例，unknown未计为失败。','', '| 臂的全名 | '+ ' | '.join(FULL[v]+suffix for v in ['V0','V1','V3'] for suffix in ['：成功率[95%CI]','：碰撞率','：超时率'])+' |','|'+'---|'*10]
 for a in arms:
  row=[LABEL[a]]
  for v in ['V0','V1','V3']:
   q=r['table'].get(v,{}).get(a)
   if not q:row+=['未进入评估']*3;continue
   s=q['success_lower'];row+=[f"{s['mean']:.2%} [{s['CI95'][0]:.2%}, {s['CI95'][1]:.2%}]",f"{q['collision']['mean']:.2%}",f"{q['timeout']['mean']:.2%}"]
  lines.append('| '+' | '.join(row)+' |')
 (HERE/'PAPER_TABLE.md').write_text('\n'.join(lines)+'\n')
 f=['| 学生全名 | 失败类型 | 数量 | 占全部局 | 占两类停滞 |','|---|---|---:|---:|---:|'];resc=['| 学生全名 | 原失败类型 | 救回臂全名 | 救回数量 |','|---|---|---|---:|']
 for v,d in r['failure_composition'].items():
  for kind,q in d.items():
   frac=q['fraction_two_stall_types'];f.append(f"| {FULL[v]} | {kind} | {q['count']} | {q['fraction_all']:.2%} | {frac:.2%} |" if frac is not None else f"| {FULL[v]} | {kind} | {q['count']} | {q['fraction_all']:.2%} | 无两类停滞 |")
   for a,n in q['rescued'].items():resc.append(f'| {FULL[v]} | {kind} | {LABEL[a]} | {n} |')
 (HERE/'FAILURE_COMPOSITION.md').write_text('\n'.join(f)+'\n\n'+'\n'.join(resc)+'\n')
 gate=read(HERE/'DISTILLATION_GATE.json');cap=read(HERE/'CAPABILITY_GATE.json');eligible=set(cap['eligible']);exclude={x['name']:x['reason'] for x in cap['excluded']};g=['| 学生全名 | 对应冻结联合MACFlow教师 | 蒸馏门 | 能力门 | 最终checkpoint |','|---|---|---|---|---|']
 for q in gate['controllers']:
  decision='通过' if q['name'] in eligible else exclude.get(q['name'],'未运行：蒸馏门未通过')
  g.append(f"| {FULL[q['variant']]} | {names[q['index']]} | {'通过' if q['valid'] else '未通过'} | {decision} | {q['checkpoint']} |")
 (HERE/'GATE_TABLE.md').write_text('\n'.join(g)+'\n')
 e=r['endpoints']
 def endpoint(k):
  q=e.get(k,{})
  if 'passes' not in q:return '证据不足'
  ci=q['CI_unknown_envelope'];return f"{'通过' if q['passes'] else '未通过'}，配对差的保守95%置信区间为[{ci[0]:.2%}, {ci[1]:.2%}]"
 def composition(v):
  q=r['failure_composition'].get(v)
  return f'{FULL[v]}无可评估学生' if not q else f"{FULL[v]}的自身停滞/安全层导致停滞为{q['自身停滞']['count']}/{q['安全层导致的停滞']['count']}局"
 audit=read(HERE/'INTEGRITY_COMPARISON.json');first=f"源码完整性审计{'通过' if audit['protected_content_pass'] else '未通过'}，Basin_C1共享数据库与journal路径断言通过。闭环重标注学生（各自噪声；V1）相对原学生（离线蒸馏、各自噪声；V0）的能力提升检验{endpoint('V1_competence')}；闭环重标注加共享随机种子学生（V3）的额外提升检验{endpoint('V3_competence')}。通过两个门后，前500步使用第13组、之后第10组修正参数的时间表（S500）相对只加安全层，在两类新学生上分别{endpoint('V1_deadlock_one')}、{endpoint('V3_deadlock_one')}；相对公平调参右手规则（RHR-fair）分别{endpoint('V1_deadlock_two')}、{endpoint('V3_deadlock_two')}。失败构成为："+'；'.join(composition(v) for v in ['V0','V1','V3'])+'。timeout仅是停滞诊断，不直接称为deadlock。'
 old=(dest/'REPORT.md').read_text();tail=old.split('\n\n',1)[1]
 if (HERE/'RESOURCE_EXCEPTION_EXTERNAL_QUEUE.json').exists():tail='另有一次外部排队作业扩展导致总CPU达到19的监测事件：本轮新训练在入口检查处退出，尚未调用训练代码；仅该未开始计算的阶段在资源释放后重新提交。见RESOURCE_EXCEPTION_EXTERNAL_QUEUE.json；这不是对unknown评估局的重跑。\n\n'+tail
 body=first+'\n\n'+tail+'\n\n两道门的逐学生结果与未通过者：\n\n'+(HERE/'GATE_TABLE.md').read_text()+'\n\n论文总表（最终复核）：\n\n'+(HERE/'PAPER_TABLE.md').read_text()+'\n\n失败构成与全部七臂救回数（最终复核）：\n\n'+(HERE/'FAILURE_COMPOSITION.md').read_text();(HERE/'REPORT.md').write_text(body)
 r['final_review']=dict(utc=utc(),all_arm_rescues=True,gate_table='GATE_TABLE.md',side_by_side_table='PAPER_TABLE.md',no_new_rollouts=True);write(HERE/'results.json',r);(HERE/'HANDOFF.md').write_text((dest/'HANDOFF.md').read_text()+'\n最终报告与表格已复核。FINAL_REVIEW.receipt.json是当前交付收据；原自动交付文件保存在automated_delivery/，原DELIVERY.receipt.json保留为历史收据。原始科学参数、checkpoint、预测与标签均未更改。\n');source_inventory('imports_final_review.json');seal('FINAL_REVIEW.receipt.json',[HERE/n for n in ['REPORT.md','HANDOFF.md','results.json','PAPER_TABLE.md','FAILURE_COMPOSITION.md','GATE_TABLE.md','DELIVERY.receipt.json','INTEGRITY_COMPARISON.json']],supersedes='DELIVERY.receipt.json',original_delivery_archive='automated_delivery',no_new_rollouts=True)
if __name__=='__main__':main()
