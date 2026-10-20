"""Run after analyze.py; reports registered descriptive quantities and delivery QA."""
from core import *
from collections import Counter
r=read(HERE/'results.json');z=np.load(HERE/'LABELS.npz');y=z['success'];u=z['unknown'];arms=z['arms'].tolist()
assert int(u.sum())==r['unknown']==329
official=read(HERE/'POSTFLIGHT_OFFICIAL_CACHE.json')['summary']
assert official['total_requested']==23296
assert official['exact_reusable']+official['partial_reusable']==23296-329
assert official['genuinely_missing']==329 and official['ambiguous']==0
unknown=read(HERE/'UNKNOWN_CASES.json')
expected={(v['case']['state_uid'],v['case']['eta_uid'],v['case']['controller_uid'],v['case']['seed_key']) for v in unknown}
missing={(v['state_uid'],v['eta_uid'],v['controller_uid'],seed) for v in read(HERE/'POSTFLIGHT_OFFICIAL_CACHE.json')['details'] for seed in v['missing_seeds']}
assert missing==expected
rates=y.mean((1,3));anchor=rates[:,arms.index('S13to10')].mean()
for pair in r['pairs']:
 ix=[arms.index(a) for a in pair['arm_order']];v=rates[:,ix]
 pair['maximum_mean_schedule_gain']=float(np.max((v[:,2:]-v[:,:2].max(1)[:,None]).mean(0)))
 pair['schedule_mean_minus_anchor']=(v[:,2:].mean(0)-anchor).tolist()
 pair['any_single_direction_mean_exceeds_anchor']=bool(np.any(v[:,2:].mean(0)>anchor))
unknown=read(HERE/'UNKNOWN_CASES.json');types=Counter(v['error']['type'] for v in unknown);steps=[v['error']['step'] for v in unknown]
r['unknown_diagnostics']=dict(types=dict(types),first_error_step_min=min(steps),first_error_step_max=max(steps),before_switch=sum(v<500 for v in steps),at_or_after_switch=sum(v>=500 for v in steps))
write(HERE/'results.json',r)
lines=['## Unknown sensitivity and failure timing','',
 '下界点判：P1 5/7、P2 6/7、P3 3/7；unknown兼容的达门对数范围分别为3–6、4–7、3–7。因此最终三项均为INCONCLUSIVE；下界点判不能替代最终判定。', '',
 f"未知共329条（{100*329/23296:.3f}%），错误类型{dict(types)}；首个错误步范围{min(steps)}–{max(steps)}，切换前{sum(v<500 for v in steps)}条。所有已知非成功均为原生timeout@700，不直接称为deadlock。逐例见FAILURE_RECORDS.json与UNKNOWN_CASES.json。", '',
 '| η pair | best single direction mean gain pp | two direction means minus anchor pp |',
 '| --- | ---: | --- |']
for p in r['pairs']:
 lines.append(f"| {p['pair']} | {100*p['maximum_mean_schedule_gain']:.2f} | {[round(100*x,2) for x in p['schedule_mean_minus_anchor']]} |")
lines+=['', '上述最大平均收益是在两个已注册方向中取较大均值；results.json另保留逐controller最大收益。共享方向与其CI依下界点判选择，属于注册的样本内多数方向统计；不是部署时已知的方向。', '', '官方只读cache postflight：22967条已知标签可复用（exact+partial）；planner排除numerical_failure，故报告329条missing。逐身份核对这些missing恰好是329条已留存unknown，无真实记录缺口；ambiguous=0，绝不据此重跑。']
report=HERE/'REPORT.md';report.write_text(report.read_text()+'\n'+'\n'.join(lines)+'\n')
handoff=HERE/'HANDOFF.md';handoff.write_text(handoff.read_text()+'\nFinal delivery: all Slurm jobs succeeded. 22 worker receipts sum to23296 unique main executions; official cache postflight22967 known reusable;329 planner-missing identities exactly equal retained numerical unknown,0 actual absent records; raw/trace/DB PASS.329 CBFSolverError unknown preserved. Point P1/P2/P3 counts5/6/3; unknown bounds3–6/4–7/3–7 yield all INCONCLUSIVE. Anchor13→10 replicated +28.24pp [15.29,41.96]. Physical dedup unavailable for4607 legacy state rows; UID/content checked for all13161 rows at draw.\nReproduce delivery supplement after analyze.py with delivery_audit.py; no rollout or DB write in this supplement. Resource SIGSTOP/CONT preserved process memory when another user was running; actual compute reverted18, otherwise at most22 CPU,0 GPU. RESOURCE_GUARD_RECEIPT/log document this.\n')
running=HERE/'HANDOFF_RUNNING.md';running.write_text(running.read_text()+'\nFINAL UPDATE: rollout and finalizer completed successfully; use HANDOFF.md and POSTFLIGHT.json as final state.\n')
write(HERE/'FINAL_DELIVERY_AUDIT.json',dict(utc=utc(),status='PASS',main_rollouts=23296,unknown=329,official_cache=official,postflight_sha256=sha(HERE/'POSTFLIGHT.json'),report_sha256=sha(report),result_sha256=sha(HERE/'results.json'),supplement_sha256=sha(__file__),no_new_rollouts_in_supplement=True))
print('Delivery PASS',r['hypotheses'])
