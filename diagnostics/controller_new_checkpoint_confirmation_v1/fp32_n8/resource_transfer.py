"""Balance still-unstarted reserved tails onto currently idle CPUs."""
from core import *
import subprocess,math
prereg();labels_allowed();gate();wave=int(sys.argv[1]);jobrows=[line.split('|') for line in subprocess.check_output(['squeue','-h','-o','%u|%T|%C|%j'],text=True).splitlines()]
assert not any(u!='zhihan' and status in ['RUNNING','COMPLETING'] for u,status,cpu,name in jobrows),'Other user computing; do not expand'
active=sum(int(cpu) for u,status,cpu,name in jobrows if u=='zhihan' and status in ['RUNNING','COMPLETING']);available=23-active;assert available>0
# Metadata prefix inspection only: never read final outcome fields.
groups={}
for q in (HERE/'main/raw').glob('*.json'):
    with q.open() as f:prefix=f.read(64)
    if 'RESERVATION_' not in prefix:continue
    m=read(q);assert not any(k in m for k in ['success','outcome','termination']);groups.setdefault(m['reservation_owner'],[]).append(m)
# Recover owner's true case order from the immutable assignment plans.
assignments={};base=read(HERE/'RESOURCE_BOOST_PLAN.json')
for j in range(6):assignments[100+j]=base['cases'][j::6]
for q in sorted(HERE.glob('RESOURCE_WAVE*_PLAN.json')):
    prior=read(q)
    for j in range(prior['helpers']):assignments[prior['wave']*1000+j]=prior['cases'][j::prior['helpers']]
ordered={}
for owner,markers in groups.items():
    uids={m['case']['rollout_uid'] for m in markers};ordered[owner]=[c for c in assignments[owner] if c['rollout_uid'] in uids]
original_rows=read(HERE/'main_MISSING_CASES.json');perm=np.random.default_rng(202610131221).permutation(len(original_rows))
for j in range(17):
    original=[original_rows[int(k)] for k in perm[j::17]];remaining=[c for c in original if not (HERE/'main/raw'/f"{c['rollout_uid']}.json").exists()]
    if remaining:ordered[j]=remaining
total=sum(map(len,ordered.values()));helpers=min(available,max(1,total//40));assert helpers>0
# Water filling caps retained workloads while preserving ≥32 frontier cases.
target=next(t for t in range(32,max(map(len,ordered.values()))+1) if sum(max(0,len(m)-t) for m in ordered.values())<=helpers*t)
selected=[];audit=[]
for owner,m in ordered.items():
    take=max(0,len(m)-target);chosen=m[-take:] if take else [];selected.extend(chosen);audit.append(dict(old_owner=owner,remaining=len(m),retained=len(m)-take,transferred=take))
assert selected,'Insufficient tail to transfer safely'
plan=dict(utc=utc(),wave=wave,helpers=helpers,reservation_schema=f'RESERVATION_WAVE{wave}_NOT_EVIDENCE',cases=selected,source_groups=audit,minimum_frontier=32,CPU_before=active,CPU_after=active+helpers,CPU_limit_compute=23,orchestration=1,outcome_fields_read=False,numerical_rules_unchanged=True)
freeze(HERE/f'RESOURCE_WAVE{wave}_PLAN.json',plan);h=sha(HERE/f'RESOURCE_WAVE{wave}_PLAN.json');freeze(HERE/f'RESOURCE_WAVE{wave}.receipt.json',dict(utc=utc(),plan_sha256=h,helper_sha256=sha(HERE/'transfer_helper.py'),merger_sha256=sha(HERE/'merge.py'),user_authorization='Use all free resources when others are absent',original_gate_reused=True,runner_sha256=sha(R9/'run.py'),run_binding_sha256=sha(HERE/'run.py'),all_cases_originally_preregistered=True,original_preflight_sha256=sha(HERE/'main_CACHE_PREFLIGHT.json'),no_outcomes_read=True))
for j,c in enumerate(selected):
    path=HERE/'main/raw'/f"{c['rollout_uid']}.json"
    existing=path.exists()
    if existing:
        m=read(path);assert m['schema'].startswith('RESERVATION_') and m['case']==c
    marker=dict(schema=plan['reservation_schema'],case=c,protocol_sha256=sha(HERE/'protocol.json'),experiment_uid=EXP,reservation_owner=wave*1000+j%helpers,reservation_plan_sha256=h,reservation_plan_path=str(HERE/f'RESOURCE_WAVE{wave}_PLAN.json'),no_success_label=True)
    if existing:write(path,marker)
    else:
        with path.open('x') as f:f.write(json.dumps(marker)+'\n')
print(json.dumps(dict(wave=wave,helpers=helpers,transferred=len(selected),target=target,CPU_after=active+helpers)))
