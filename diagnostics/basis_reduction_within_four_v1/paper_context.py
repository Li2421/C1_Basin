"""Read-only historical paper context; no new rollout or parameter selection."""
import guard
from guard import HERE,ROOT
from pathlib import Path
import json,sqlite3,hashlib,datetime,statistics

def read(path):return json.loads(Path(path).read_text())
def dump(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def mean(xs):return statistics.mean(xs)if xs else None

def physical_key(state):
 return tuple(tuple(tuple(float(v).hex()for v in p)for p in state[key])for key in('positions','velocities','goals'))

def historical():
 path=HERE/'inputs/paper_context/HISTORICAL_ROWS.json'
 if path.exists()and read(path).get('physical_world_match'):return read(path)
 comparison=read(HERE/'inputs/paper_context/COMPLETED_COMPARISON.json');selected=read(HERE/'REPRESENTATIVE17_FROZEN.json')['selected'];levels={x['type_id']:x['level']for x in selected};scenes={(x['type_id'],x['level']):x['scene']for x in read(HERE/'SCENES.json')['entries']};profiles=[]
 con=sqlite3.connect('file:'+str(ROOT/'shared_rollout_db/rollout.sqlite')+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
 for profile in comparison['completed_paper_profiles']:
  if'reference_default'not in profile['labels']:continue
  records=[]
  for uid in profile['rollout_uids']:
   row=con.execute('SELECT r.state_uid,r.conflict_quarantined,s.physical_state_json,sc.metadata_json AS physical_world_json FROM rollout r JOIN state s ON r.state_uid=s.state_uid JOIN scenario sc ON s.scenario_uid=sc.scenario_uid WHERE r.rollout_uid=?',(uid,)).fetchone();assert row and not row['conflict_quarantined']
   source=con.execute('SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf ON rs.source_uid=sf.source_uid WHERE rs.rollout_uid=? LIMIT 1',(uid,)).fetchone();assert source
   with Path(source['path']).open()as f:r=next(json.loads(line)for i,line in enumerate(f,1)if i==source['source_line'])
   t=int(r['type_id']);si=int(r['state_index']);level=levels[t];assert 0<=si<8
   assert physical_key(json.loads(row['physical_state_json']))==physical_key(scenes[(t,level)]['states'][si]),('Paper physical-state mismatch',profile['id'],t,si)
   world=json.loads(row['physical_world_json']);expected=scenes[(t,level)]
   assert{k:v for k,v in world['config'].items()if k!='scene'}=={k:v for k,v in expected['config'].items()if k!='scene'},('Paper physical-config mismatch',profile['id'],t,si)
   assert world['walls']==expected['walls']and world['plant']=='frozen_swap_step',('Paper physical-geometry mismatch',profile['id'],t,si)
   records.append(dict(identity=[t,level,si],rollout_uid=uid,success=r['success'],outcome=r['outcome'],completion_time=r['completion_time'],truncated_time=r['truncated_time'],trace=r['trace'],source_file=source['path'],source_line=source['source_line']))
  assert len(records)==136 and len({tuple(r['identity'])for r in records})==136
  profiles.append(dict(id=profile['id'],name=profile['name'],options=profile['options'],overall=profile['overall'],single_robot_passed_scenes=profile['single_robot_passed_scenes'],records=records))
 con.close();data=dict(profiles=profiles,physical_state_match=True,physical_world_match=True,physical_states=136,states=[0,7],new_environment_steps=0,reference_configs_are_not_all_verified_author_defaults=True);dump(path,data);return data

def refresh():
 history=historical();policies=[]
 combo_names={c['code']:c['name']for c in read(HERE/'COMBINATIONS.json')['combinations']}
 for file in sorted(HERE.glob('FIXED_*_screen.json')):
  data=read(file);code=data['code']
  for controller,label in [('centralized_sampling','集中式随机采样模型预测控制'),('gradient_mpc','梯度下降模型预测控制')]:
   records=[r for r in data['records']if r['identity'][2]==controller and r['identity'][3]<8];known=[r for r in records if r['result']is not None]
   if len(known)!=136:continue
   times=[r['result']['completion_time']for r in known if r['result']['success']];mapping={(r['identity'][0],r['identity'][1],r['identity'][3]):r['result']for r in known};paired=[]
   for profile in history['profiles']:
    old={tuple(r['identity']):r for r in profile['records']};keys=sorted(mapping.keys()&old.keys());assert len(keys)==136;common=[k for k in keys if mapping[k]['success']and old[k]['success']]
    paired.append(dict(paper=profile['id'],physical_cases=136,common_success=len(common),C1_only=sum(mapping[k]['success']and not old[k]['success']for k in keys),paper_only=sum(old[k]['success']and not mapping[k]['success']for k in keys),common_success_C1_mean=mean([mapping[k]['completion_time']for k in common]),common_success_paper_mean=mean([old[k]['completion_time']for k in common])))
   policies.append(dict(code=code,nominal=controller,nominal_name=label,cases=136,success=len(times),success_mean_time=mean(times),truncated_mean_time=mean([r['result']['truncated_time']for r in known]),paired=paired))
 out=dict(time=datetime.datetime.now().astimezone().isoformat(),source='inputs/paper_context/COMPLETED_COMPARISON.json',historical_profiles=[{k:v for k,v in p.items()if k!='records'}for p in history['profiles']],current_fixed_policies=policies,scope='Same136 physical scene/state0..7; each C1 nominal compared separately to paper native fixed policy. Does not cover independent states8..11. Oracle results are excluded from fixed-policy ranking.',limitations=read(HERE/'inputs/paper_context/COMPLETED_COMPARISON.json')['limitations'],used_for_basis_selection=False,new_rollouts=0)
 lines=['# 论文端口历史对照（只读复用）\n','同17个场景、状态0–7共136个物理初态；初始位置、速度、目标、半径、墙体、速度上限、步长和时限已逐条核验一致。论文使用自己的控制器与安全求解器；C1两个名义控制器分别比较。\n','下列为此前已跑的参照配置，不把所有数值冒称作者默认。已知端口异常和单机诊断限制尚未修复；本表不用于宣称原论文方法优劣，也不参与本轮删减判据。\n','| 论文方法 | 成功/136 | 成功条件均时·秒 | 全队列截断均时·秒 | 原生异常局数 | 单机通过/17 |','|---|---:|---:|---:|---:|---:|']
 for p in out['historical_profiles']:
  x=p['overall'];lines.append(f"| {p['name']} | {x['success']} | {x['success_mean_time']:.2f} | {x['full_completed_cohort_truncated_mean_time']:.2f} | {x['original_first_error_unknown']} | {p['single_robot_passed_scenes']} |")
 lines+=['\n表中成功数采用此前统一异常处理协议：受影响机器人在出错当步动作置零，后续步骤重新尝试；原生首错即未知的局数另列。一次对局可能既出现原生异常又在继续运行后成功，因此两列不能相加划分结局。',
         '\n新组合的Oracle是逐案例有限搜索上限，不能与上述固定参数策略混称部署成功率。下面只列已完成的C1固定参数；共同成功案例的配对时间和逐配置参数保存在PAPER_CONTEXT.json。','\n| C1组合 | 名义控制器 | 固定参数成功/136 | 成功条件均时·秒 |','|---|---|---:|---:|']
 for p in policies:lines.append(f"| {combo_names.get(p['code'],p['code'])} | {p['nominal_name']} | {p['success']} | {p['success_mean_time']} |")
 if not policies:lines.append('| 尚未完成新组合固定参数 | — | — | — |')
 lines+=['\n本表不覆盖独立状态8–11；它们虽然与两个控制器合计也为136例，却不是这里的136个物理初态。论文端口保持原参数、原实现，本轮没有新增论文rollout。']
 out['report_lines']=lines;dump(HERE/'PAPER_CONTEXT.json',out);(HERE/'PAPER_CONTEXT.md').write_text('\n'.join(lines)+'\n');return out

if __name__=='__main__':
 d=refresh();print('Historical paper profiles',len(d['historical_profiles']),'current fixed policies',len(d['current_fixed_policies']),'new rollouts0')
