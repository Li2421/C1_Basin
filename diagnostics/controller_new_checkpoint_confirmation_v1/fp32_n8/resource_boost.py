"""Reserve only distant, unstarted tail cases; no success-label inspection."""
from core import *
import math
prereg();labels_allowed();gate();rows=read(HERE/'main_MISSING_CASES.json');perm=np.random.default_rng(202610131221).permutation(len(rows));parts=[[rows[int(i)] for i in perm[j::17]] for j in range(17)]
missing=[[r for r in part if not (HERE/'main/raw'/f"{r['rollout_uid']}.json").exists()] for part in parts];target=math.ceil(sum(map(len,missing))/23);selected=[];parts_audit=[]
for i,m in enumerate(missing):
    take=max(0,min(len(m)-target,len(m)-32));chosen=m[-take:] if take else [];selected.extend(chosen);parts_audit.append(dict(original_worker=i,remaining_at_snapshot=len(m),reserved=take,retained=len(m)-take,minimum_unreserved_front_buffer=32))
assert selected
plan=dict(utc=utc(),total_requested_unchanged=12288,labels_read=False,source_worker_count=17,helper_count=6,total_compute_CPU=23,orchestration_CPU=1,GPU=0,original_assignment_sha256=sha(HERE/'main_MISSING_CASES.json'),protocol_sha256=sha(HERE/'protocol.json'),cases=selected,parts=parts_audit,policy='Explicit non-evidence reservations prevent originals executing helper-owned tail cases. No in-flight case reserved; ≥32 not-yet-written front cases retained at snapshot. No success/outcome fields in markers; markers replaced atomically by real completed records. Merger skips validated operational markers; endpoint waits both original and helper jobs.')
freeze(HERE/'RESOURCE_BOOST_PLAN.json',plan)
freeze(HERE/'RESOURCE_BOOST.receipt.json',dict(utc=utc(),user_authorization='If nobody else uses resources, use all available resources',Slurm_other_users_running=0,CPU_alloc_before=17,CPU_available=24,compute_after=23,orchestration=1,plan_sha256=sha(HERE/'RESOURCE_BOOST_PLAN.json'),frozen_numerical_rules_unchanged=True,run_binding_sha256=sha(HERE/'run.py'),runner_sha256=sha(R9/'run.py'),helper_sha256=sha(HERE/'helper.py'),merger_sha256=sha(HERE/'merge.py'),no_outcomes_read=True))
for j,c in enumerate(selected):
    dest=HERE/'main/raw'/f"{c['rollout_uid']}.json";marker=dict(schema='RESERVATION_NOT_EVIDENCE',case=c,protocol_sha256=sha(HERE/'protocol.json'),experiment_uid=EXP,reservation_owner=100+j%6,reservation_plan_sha256=sha(HERE/'RESOURCE_BOOST_PLAN.json'),no_success_label=True)
    with dest.open('x') as f:f.write(json.dumps(marker)+'\n')
print(json.dumps(dict(reserved=len(selected),remaining_at_snapshot=sum(map(len,missing)),target=target,CPU=23)))
