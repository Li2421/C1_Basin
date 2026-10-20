"""Part C: only read historical records; never invoke a controller/environment."""
from common import *
import collections,difflib,csv
R36=D/'axis4b_agent_count_v1';GAP=D/'gap_flow_competence_v3_recovery'
def main():
 prereg();cases=[c for c in read(R36/'pilot_CASES.json') if c['N']==2 and c['kind']=='Flow'];assert len(cases)==48;groups={};records=[];asset=read(R36/'ASSET_AUDIT.json')['assets']['2'];assert sha(asset['checkpoint'])==asset['checkpoint_sha256']
 for arm in ['safety_baseline','F10','S13t2']:
  rr=[]
  for c in sorted([c for c in cases if c['arm']==arm],key=lambda c:c['state_index']):
   f=R36/'pilot/raw'/f"{c['rollout_uid']}.json";q=read(f);assert q['flow_checkpoint_sha256']==asset['checkpoint_sha256'];assert sha(q['trace_path'])==q['trace_sha256'];assert q['case']==c and q['jax_enable_x64']==False and q['deadlock']==0
   with np.load(q['trace_path']) as z:
    assert len(z['positions'])-1==q['physical_steps'];assert np.array_equal(z['positions'][1:],z['positions'][:-1]+.05*z['actions']);assert np.asarray(q['first_action'],np.float64).tobytes()==z['actions'][0].astype(np.float64).tobytes()
   rr.append(q);records.append(dict(arm=arm,state_index=c['state_index'],state_uid=c['state_uid'],original_uid=q['original_uid'],success=q['success'],outcome=q['outcome'],steps=q['physical_steps'],raw_path=str(f),raw_sha256=sha(f)))
  counts=dict(collections.Counter(q['outcome'] for q in rr));groups[arm]=dict(states=16,successes=sum(q['success'] for q in rr),success_pct=sum(q['success'] for q in rr)/16*100,termination_counts=counts,success_state_uids=[q['state_uid'] for q in rr if q['success']],success_state_indices=[q['case']['state_index'] for q in rr if q['success']])
 assert [groups[a]['successes'] for a in ['safety_baseline','F10','S13t2']]==[1,2,3]
 before=read(R36/'CONSISTENCY_GATE_BEFORE_ACTION_ENCODING_FIX.json');after=read(R36/'CONSISTENCY_GATE.json');frozen=read(R36/'GATE_IMPLEMENTATION_FROZEN.receipt.json');hashchecks={}
 for f,h in frozen['files'].items():hashchecks[f]=sha(f)==h;assert hashchecks[f]
 gate=[]
 for c in read(R36/'gate_CASES.json'):
  q=read(R36/'gate/raw'/f"{c['rollout_uid']}.json");assert q['execution_code_sha256']==sha(R36/'runtime.py');assert sha(c['reference_trace'])==c['reference_trace_sha256']
  if c['kind']=='expert':continue
  with np.load(q['trace_path']) as new,np.load(c['reference_trace']) as old:
   p0=new['positions'];p1=old['positions'];first=old['u_safe'][0];ok=p0.dtype==p1.dtype and p0.shape==p1.shape and p0.tobytes()==p1.tobytes() and np.asarray(q['first_action'],np.float64).tobytes()==first.astype(np.float64).tobytes() and q['outcome']==c['reference_termination'] and q['episode_length']==c['reference_steps'];assert ok;gate.append(dict(N=c['N'],rollout_uid=c['rollout_uid'],passed=ok,first_action_recording_dtype=str(first.dtype)))
 assert len(gate)==16 and before['primitive_passed'] and after['primitive_passed'];assert not before['expert_passed'] and after['expert_passed'];assert all(v['positions_bytes'] and v['terminal'] and v['steps'] for v in before['checks']['expert']);assert all(v['passed'] for v in after['checks']['expert'])
 oldchecker=R36/'check_gate_before_action_encoding_fix.py';newchecker=R36/'check_gate.py';diff=''.join(difflib.unified_diff(oldchecker.read_text().splitlines(True),newchecker.read_text().splitlines(True),fromfile=str(oldchecker),tofile=str(newchecker)));freeze=read(GAP/'frozen/freeze_manifest.json');hist={};paths=[GAP/'dev_50k/summary.json',GAP/'acceptance_basic/summary.json',GAP/'acceptance_release_even_v2/summary.json',GAP/'acceptance_release_odd_v2/summary.json',GAP/'frozen_eval/opposing/summary.json',GAP/'replication_n2/opposing/summary.json']
 for f in paths:
  q=read(f);rr=q.get('rollouts',[]);hist[str(f)]=dict(sha256=sha(f),mode=q.get('mode'),N=q.get('N'),checkpoint_sha256=q.get('checkpoint_sha256'),count=len(rr),successes=sum(bool(x.get('success')) for x in rr),termination_counts=dict(collections.Counter(x.get('termination') for x in rr)),scope='Use source mode/selection metadata; do not mix future samples or checkpoint identities')
 out=dict(utc=utc(),conclusion='门的执行正确，N=2 Flow 在相向任务上能力不足',scope='Original fixed16pilot, three predeclared arms andknownsuccess>=20% rule only; not universal incapability ofthis policy/eta family',arms=groups,records=records,gate_primitive_reverified=gate,checker_diff=diff,frozen_execution_hashes_unchanged=hashchecks,expert_fix=read(R36/'ACTION_RECORDING_SCHEMA_REPAIR.json'),checkpoint_freeze=freeze,historical_evidence=hist,no_rollouts=True,not_rejudged=True,whole_TT_historical_gate='Unavailable full historical TT trajectory; only the sixteen primitive Flow/single-safety/native paths were historical-byte-reproduced. New TT composition was separately unit-tested. This audit does not invent whole-TT trace equivalence.')
 write(HERE/'N2_GATE_AUDIT.json',out)
 with (HERE/'N2_GATE_EPISODES.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 md=out['conclusion']+'。本结论限于原16个相向pilot与原三臂，门仍为20%，18.75%不四舍五入成20%，没有重判门或新增rollout。\n\n| 臂 | 成功 | 碰撞 | timeout | unknown | 成功state索引 |\n|---|---|---|---|---|---|\n'
 for a,v in groups.items():md+=f"| {a} | {v['successes']}/16 ({v['success_pct']:.2f}%) | {v['termination_counts'].get('collision',0)} | {v['termination_counts'].get('timeout',0)} | {v['termination_counts'].get('numerical_failure',0)} | {v['success_state_indices']} |\n"
 md+='\n逐条state_uid、original_uid、终止、步数、raw路径及hash见N2_GATE_EPISODES.csv与JSON；三个臂的成功state可重叠，不把不同臂当独立state。\n\n冻结checkpoint '+asset['checkpoint_sha256']+' 的选择依据明确为非相向navigation四类各12/12，时间释放even12/12、odd11/12，冻结前未打开相向测试。训练数据与选择/验收分开：含特权示范不代表相向闭环控制已通过。后续相向frozen_eval与replication的原记录也在本审计引用，不能把非相向验收当相向成功保证。本轮pilot所有机器人同时对向争用同一Gap，且使用预注册R10精确η菜单、相应TT链和新state/RNG；与历史非相向/单向/分时放行任务不同。\n\n修正的是专家复现检查器：旧actions是float32存储，旧checker先提升为float64再与新原生64位动作比较；新checker把新动作按旧artifact dtype序列化后比较。未放宽容差、未改变物理或执行精度、未重跑gate。历史专家原生float64首动作未存储，不能宣称那些未记录字节已验证。完整float64位置、终止与步数在修复前已一致。Flow primitive在修复前后一直16/16通过，本次只读重新逐字节核对16条首动作/完整位置/终止/步数，包含N2的4条。冻结runtime/controllers及全部gate原始执行hash未改变。\n\n“实际路径与历史一致”仅限可一一对应的16条Flow→单安全投影→native历史复现路径。新TT修正菜单的完整历史轨迹本来不存在，不能把该门扩写成全部pilot/全部TT轨迹逐位复现；TT组合顺序另有原UNIT_TESTS证据。这一证据范围限制不是发现执行或判定错误。原门只按known成功比例判定、unknown给界；N2三臂无unknown，最大3/16，结论无需改门。\n\n检查器完整diff、各源路径/sha、未变执行hash与历史任务统计见N2_GATE_AUDIT.json。\n'
 (HERE/'N2_GATE_AUDIT.md').write_text(md);seal('N2_GATE_AUDIT.receipt.json',[HERE/'N2_GATE_AUDIT.json',HERE/'N2_GATE_EPISODES.csv',HERE/'N2_GATE_AUDIT.md'],read_only=True,no_rollouts=True);print(out['conclusion'],groups,flush=True)
if __name__=='__main__':main()
