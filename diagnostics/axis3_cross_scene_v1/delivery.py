from common import *
import csv,subprocess

def call(script,args,log,python=None):
 with (HERE/log).open('w') as f:subprocess.run([python or sys.executable,'-B',str(HERE/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def textfile(name,text):(HERE/name).write_text(text)
def cell(v):
 if not v:return 'N/A（见缺口）'
 a,b=v['success_pct'];l,h=v['success_95CI_pct'];return (f'{a:.2f}' if abs(a-b)<1e-10 else f'{a:.2f}–{b:.2f}')+f' [{l:.2f}, {h:.2f}]'
def main():
 call('aggregate.py',[],'AGGREGATE.log');call('fewshot.py',[],'FEWSHOT.log');plotpy='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python';call('plots.py',[],'PLOTS.log',plotpy);results=read(HERE/'results.json');few=read(HERE/'FEWSHOT_RESULTS.json');begin=read(HERE/'INTEGRITY_BEGIN.json');end=read(HERE/'INTEGRITY_END.json');audit=read(HERE/'INTEGRITY_COMPARISON.json');details={};changed=[]
 for section in audit['changed_sections']:
  a=begin[section];b=end[section]
  if section in ['Basin','MACFlow']:
   details[section]={k:dict(before=a.get(k),after=b.get(k)) for k in a if a.get(k)!=b.get(k)}
   for k in ['HEAD','diff_sha256']:
    if a.get(k)!=b.get(k):changed.append(f'- {section}/{k}: {a.get(k)} → {b.get(k)}')
   for line in sorted(set(b['status'].splitlines())-set(a['status'].splitlines())):changed.append(f'- {section} status added: {line}')
   for line in sorted(set(a['status'].splitlines())-set(b['status'].splitlines())):changed.append(f'- {section} status removed: {line}')
  elif isinstance(a,dict) and isinstance(b,dict):
   details[section]={k:dict(before=a.get(k),after=b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
   for k,v in details[section].items():changed.append(f'- {section} {k}: {v["before"]} → {v["after"]}')
  else:details[section]=dict(before=a,after=b);changed.append(f'- {section}: {a} → {b}')
 write(HERE/'INTEGRITY_DETAILS.json',details);methods=['nominal only','safety baseline','F10','S500','A3 zero-shot','A0','G_E','best fixed oracle','full menu oracle','expert','LOSO critic','source mean action','Ring default schedule'];table=[];metric=[];gaps=[]
 for sc in SCENES+['pooled']:
  r=results[sc];mm=r.get('methods',{});row=dict(subaxis=sc,evidence=r.get('evidence','GAP'))
  for name in methods:
   row[name]=cell(mm.get(name))
   if not mm.get(name):gaps.append(dict(scene=sc,method=name,reason=r.get('reason','R16 frozenA3/A0/G_E have Ring-specific physical23/PCA/menuonehot, incompatible with24-dimensional LOSO critic; no matched expert labels on new32states'),required='Compatible scene-independent frozen model or separately designed privileged reference; never fabricate padding/adaptation'))
  table.append(row)
  for name,v in mm.items():metric.append(dict(scene=sc,method=name,evidence=r.get('evidence','GAP'),**v))
 with (HERE/'AXIS_TABLE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['subaxis']+methods+['evidence']);w.writeheader();w.writerows(table)
 cols=['subaxis']+methods+['evidence'];textfile('AXIS_TABLE.md','| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(str(row[c]) for c in cols)+' |' for row in table)+'\n\n% [95% CI]，范围为unknown识别界。S500列为按H缩放的S13_t2（Ring默认）；Toy缺口保留。pooled只对可评估场景等权，不冒充四场景。标准Ring critic不做无依据的输入转换。\n');write(HERE/'METHOD_METRICS.json',metric)
 with (HERE/'METHOD_METRICS.csv').open('w',newline='') as f:
  cols=['scene','method','evidence','success_pct','success_95CI_pct','mean_regret_pp','max_regret_pp','catastrophes_certain','catastrophes_possible','harm_certain','harm_possible','events_pct'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items() if k in cols} for row in metric)
 gaps.extend([dict(scene='Toy',reason='OriginalR13/R20 FP64 versus mandated newFP32:0/16 fulltrajectorygate. No newpilot/probe/eval permitted; old sourceFP64 labels retained withpartial3/5 R20descriptor',required='An explicitly authorized precision policy compatible with historical execution, or an independently validated FP32 source; no source edits'),dict(scene='XS1 pooled',reason='Four-target aggregate unavailable whenToy has no prospective labels',required='Resolve Toy reproduction gap; current available3target result is partial'),dict(scene='source descriptors',reason='Toy source isotropic has only3/5 cachedR20quantities, last2 source-fold mean imputed; source affine operators shareonecheckpoint',required='Compatible full-duration unlabeled measurements after compliantgate; no claim of independentToy/DB/Ring checkpoints'),dict(scene='source action coverage',reason='R19 selectedFour oldlabels lackS13_t1; no fabricatedlabels. Source means include available compatiblecontroller/scene coverage only, retained in FROZEN_CHOICES.json',required='Additional compatible historicallabels if found; otherwise explicit coverage limitation')]);write(HERE/'GAPS.json',gaps)
 pooled=results['pooled'];xs1=pooled['XS1'];xs2=pooled['XS2'];sentences=[]
 for sc in SCENES:
  r=results[sc]
  if not r.get('methods'):sentences.append(sc+'：精度复现门未通过，前瞻评估缺口')
  else:
   regret=r['XS2']['regret_pp'];gain=r['paired']['XS5_best_correction_minus_safety']['CI_pp'];sentences.append(f'{sc}：LOSO遗憾{regret}pp，最佳修正−safety CI={gain}pp')
 first=f"完整性审计{'通过' if audit['passed'] else '未完全通过（差异清单附后）'}。XS1 {xs1['status']}：可评估{pooled['target_scenes_available']}/4场景的critic−源均值遗憾差CI={xs1['CI_pp']}pp（可用子集判定{xs1['available_scene_status']}，不代替四场景终点）；XS2 {xs2['status']}：确定≤5pp场景数{xs2['definite_scene_count']}，含缺失场景可能数{xs2['possible_scene_count_including_missing']}，门槛3。"+'；'.join(sentences)+'。'
 body='\n\n复现门：DB/Four/Ring各16/16逐位通过，Toy0/16。Toy旧执行FP64与共享协议FP32冲突，没有改用FP64强行过门；原Toy旧标签仅作其他target的历史源数据。所有新执行x64=False。源Toy仅isotropic有可复用的R20三项描述，另外两项缺失以源折训练均值插补，四场景完整迁移结论受此覆盖限制。\n\nDB/Ring各3个历史affine算子共用各自1个checkpoint，不能写成各3个独立训练政策；Four选定3配方best/final共6真实checkpoint。12控制器全部能力门保留。新pilot384、无标签probe384、main6912；newstate generation176+Fourprobe补充16，全部按原生成器、配置和UID/可恢复物理三元组去重。未知历史物理记录未能恢复的范围见去重审计。DB只通过几何reset有效性，没有把它等同联合可解。\n\nLOSO四target模型各3fit，内层留一源场景CV，全部拟合、预处理和step只用其他3场景的旧标签；目标scene在train/CV中行数均0。源码目标排除与输入hash收据在各MODEL_FROZEN、TRAINING_INPUTS_FROZEN和SOURCE_LABEL_PROVENANCE中。每个控制器的选择和分析实现均在任何main标签前冻结；新pilot只作事前能力门，不训练或调参。每fold所选step及源均值动作、unknown可能动作与缺臂覆盖见FROZEN_CHOICES.json。\n\nXS3、XS5为描述性：交叉拟合时间表−固定、最佳修正−safety及其CI分别见逐场景paired字段。oracle按原state索引奇偶分半选择/评估再交换，bootstrap每次重新选择且保留原奇偶；同时报告样本内上界。CI5000次controller→state簇抽样，两seed与各臂保持配对；合并行固定场景等权，未外推到未测试拓扑。source mean action主表为冻结的确定选择，XS1敏感性包络另覆盖旧源unknown允许的动作。unknown上下界保留，不重跑、不当作已知失败。nativeDB deadlock与timeout分别保存；没有从停滞或timeout推断deadlock。oracle事件率为下界训练的确定性首选臂，选择不确定性体现在成功率/遗憾包络。\n\nXS4：前16与后16原state_uid排序分离校准/评估，k=1/2/4/8/16每臂已知episode，无放回1000次模拟，严格沿用R23经验法E和Beta阈值0.9安全法S；默认D0为Ring时间表。FEWSHOT_RESULTS.json同时给均值遗憾≤3pp的kstar和所有controller的MC可靠性≥0.9的kstar，未达网格只报告>16/arm、>112已知标签，不声称更大预算必成功。unknown不进入校准池，实际查询成本可能大于已知标签预算。fewshot曲线PDF/PNG/SVG用既有绘图环境输出，无包安装。\n\n实现/资源偏差完整保留：初版复现索引遗漏scene过滤，8条有效轨迹复用，错误初始化在任何物理步前退出，修正后只补56条，见SCENE_INDEX_REPAIR.json；原错误batch与64条计划预检保留。探测窗口最初误记R20为200步，读精确历史CASES后在新probe前封存PROBE_DURATION_AMENDMENT，改按3H/7；R19b200步旧trace不冒充长窗口，补充预定Fourprobe16states，没有改controller、阈值或抽样种子。旧标签仍复用。下游脚本尚未就绪时编排退出，原ERROR文件保留；只续执行未完成阶段，无rollout/probe/unknown重跑。\n\n首次GPU请求误写gpu:1，短时保留整卡；已取消并改为两个shard:1作业，6个完成CV拟合在种子/行数/transform/checkpoint逐项一致后复用，部分拟合原样归档，没有根据性能调参，见GPU_GRES_REPAIR.json。正确GPU作业依赖其他已排队GPU任务结束；main在其他任务占满18CPU时按MAIN_RESOURCE_DEFERRED等待空位，main阶段外部又新增2CPU，而初版监控已在阶段空隙退出，重启监控前总占用曾短时20；已暂停本轮2分片恢复实际18，并修正为持续监控到本轮数值完成，见RESOURCE_MONITOR_RESTART.json与RESOURCE_STATUS。原错误资源请求与响应间隙不隐瞒为全程合规。\n\n官方DB流程：登记用connect/transaction仅INSERT OR IGNORE新身份，旧行hash逐表前后一致；每阶段提交前preflight，worker只写默认官方journal，单进程官方merger与postflight，unknown用直接UID核查存在而不重跑。transport三元组规范化保留原uid/key_for随机流，不改旧row，不以新编码宣布旧标签失效。所有probe仅SUMMARY_ONLY，不生成task标签。\n\n文件：AXIS_TABLE.md/csv、METHOD_METRICS.json/csv、results.json、FEWSHOT_RESULTS.json/curves、GAPS.json、各模型checkpoint、复现/能力/特征/选择冻结、DB登记/预检/后检、INTEGRITY起止与比较、INDEPENDENT_DELIVERY_REVIEW.json、HANDOFF.md。本轮无视频要求；无旧源码修改、环境安装或git写操作。\n'
 textfile('REPORT.md',first+body+('\n完整性差异：\n\n'+'\n'.join(changed)+'\n' if changed else '\n起止审计一致。\n'));textfile('HANDOFF.md',first+'\n\n指定Python与工作根目录见PREREGISTRATION。数值完成须NUMERICAL_COMPLETE.receipt.json存在且机械review通过；严格源码清单审计另看首段，不能与机械核验混淆。复现v2与train_v2为有效入口；原train.py请求整GPU仅作归档，勿执行。已完成rollout与unknown不重跑。三scene可评估、Toy精度缺口、源Toy3/5特征和Four源缺臂必须在论文中保留。全部数据留在本目录，只有官方DB例外。下一步完整读取round35并按顺序执行，不并行未完成轮次。\n')
 print('Round34 tables/report delivered',flush=True)
if __name__=='__main__':main()
