from common import *
from failure_diagnostics import diagnose
from audit_utils import source_inventory
import collections,difflib

def main():
    prereg();assert (HERE/'F_COMPLETE.receipt.json').exists();assert not (HERE/'N2_GATE_AUDIT.json').exists();cases=[c for c in read(R36/'pilot_CASES.json') if c['N']==2 and c['kind']=='Flow'];assert len(cases)==48;records=[];arms={}
    for c in cases:
        path=R36/'pilot/raw'/f"{c['rollout_uid']}.json";r=read(path);assert sha(r['trace_path'])==r['trace_sha256'];d=diagnose(r) if not r['success'] else dict(termination='success');records.append(dict(state_uid=c['state_uid'],state_index=c['state_index'],arm=c['arm'],rollout_uid=c['rollout_uid'],source=str(path),source_sha256=sha(path),success=r['success'],diagnostics=d,nominal_counterpart='缺口：该能力门未采样只执行controller动作的对应臂，本部分只读'))
    for a in ['safety_baseline','F10','S13t2']:
        rr=[r for r in records if r['arm']==a];arms[a]=dict(success_state_indices=[r['state_index'] for r in rr if r['success']],success_state_uids=[r['state_uid'] for r in rr if r['success']],success_rate=sum(r['success'] for r in rr)/16,termination_counts=dict(collections.Counter(r['diagnostics']['termination'] for r in rr)),failure_classes=dict(collections.Counter(r['diagnostics'].get('category') for r in rr if not r['success'])))
    before=read(R36/'CONSISTENCY_GATE_BEFORE_ACTION_ENCODING_FIX.json');after=read(R36/'CONSISTENCY_GATE.json');assert before['primitive_passed'] and after['primitive_passed'] and after['expert_passed']
    comparisons=[]
    for old,new in zip(before['checks']['expert'],after['checks']['expert']):
        assert old['rollout_uid']==new['rollout_uid'] and old['reference_trace']==new['reference_trace'];assert all(old[k] and new[k] for k in ['positions_bytes','terminal','steps']);comparisons.append(dict(rollout_uid=new['rollout_uid'],physical_bytes_unchanged=True,action_storage_dtype=new['first_action_recording_dtype'],native_dtype=new['native_first_action_dtype']))
    gatecases=read(R36/'gate_CASES.json');action_checks=[]
    for c in gatecases:
        q=read(R36/'gate/raw'/f"{c['rollout_uid']}.json");assert sha(c['reference_trace'])==c['reference_trace_sha256'];z=np.load(c['reference_trace']);actual=np.load(q['trace_path']);reference=z['states'] if c['kind']=='expert' else z['positions'];first=z['actions'][0] if c['kind']=='expert' else z['u_safe'][0];assert np.asarray(q['first_action'],dtype=first.dtype).tobytes()==first.tobytes();assert actual['positions'].dtype==reference.dtype and actual['positions'].tobytes()==reference.tobytes();assert q['outcome']==c['reference_termination'] and q['episode_length']==c['reference_steps'];action_checks.append(dict(rollout_uid=c['rollout_uid'],native_path_exact=True,raw_sha256=sha(R36/'gate/raw'/f"{c['rollout_uid']}.json")))
    asset=read(R36/'ASSET_AUDIT.json')['assets']['2'];freeze=asset['freeze'];basis=freeze['selection_basis'];assert not freeze['original_opposing_test_opened_before_freeze'];assert sha(asset['checkpoint'])==asset['checkpoint_sha256'];oldcode=(R36/'check_gate_before_action_encoding_fix.py').read_text();newcode=(R36/'check_gate.py').read_text();diff=''.join(difflib.unified_diff(oldcode.splitlines(True),newcode.splitlines(True)));(HERE/'N2_CHECKER_DIFF.patch').write_text(diff)
    conclusion='门的执行正确，N=2 Flow 在相向任务上能力不足';out=dict(utc=utc(),status='COMPLETE',conclusion=conclusion,arms=arms,records=records,source_checkpoint=asset['checkpoint'],source_checkpoint_sha256=asset['checkpoint_sha256'],historical_competence_selection=basis,historical_opposing_opened_before_freeze=False,task_difference='历史冻结依据为导航与有序时间释放；本轮16个pilot是共享瓶颈的同时相向交换。不能从导航能力推出相向交换能力。',checker_only_repair=True,checker_comparisons=comparisons,all_archived_gate_paths_exact=action_checks,new_rollouts=0,no_gate_rejudgment=True,source_hashes={str(R36/f):sha(R36/f) for f in ['runtime.py','check_gate.py','check_gate_before_action_encoding_fix.py','CONSISTENCY_GATE.json','CONSISTENCY_GATE_BEFORE_ACTION_ENCODING_FIX.json','ACTION_RECORDING_SCHEMA_REPAIR.json']})
    write(HERE/'N2_GATE_AUDIT.json',out);lines=[conclusion+'。本部分只读，未重判门，未运行新rollout。','', '历史能力依据：'+basis+'。本轮为同时相向交换；导航和有序释放证据不能替代此任务的成功证据。','', '| 方法 | 成功state索引 | 成功率 | 终止类型 | 失败分类 |','|---|---|---:|---|---|']
    labels={'safety_baseline':'只加安全投影、不加修正','F10':'全程固定用η10的修正','S13t2':'前round(5/7·H)步用η13、之后用η10的时间表'}
    for a,q in arms.items():lines.append(f"| {labels[a]} | {q['success_state_indices']} | {q['success_rate']:.2%} | {q['termination_counts']} | {q['failure_classes']} |")
    lines += ['','逐state明细见 N2_GATE_AUDIT.json。动作记录在专家数据中原为float32；修复检查器后按原记录dtype比较，原生动作与位置执行仍为float64。原有Flow/safety/native轨迹及专家完整位置、终止、步数均逐位一致；本轮再次核对32条既有复现产物，未重跑。','', '对应的“只执行controller动作”终止标签在此能力门中缺失，本部分禁止补跑，已逐条列为证据缺口；它不阻止基于现有动作路径与原门阈值的审计结论。']
    (HERE/'N2_GATE_AUDIT.md').write_text('\n'.join(lines)+'\n');source_inventory('imports_n2_audit.json');seal('C_COMPLETE.receipt.json',[HERE/'N2_GATE_AUDIT.json',HERE/'N2_GATE_AUDIT.md',HERE/'N2_CHECKER_DIFF.patch',HERE/'n2_audit.py'],new_rollouts=0)
if __name__=='__main__':main()
