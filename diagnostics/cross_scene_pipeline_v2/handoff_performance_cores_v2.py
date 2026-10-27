"""Checkpoint only active E-core jobs; Slurm reallocates free P cores normally."""
from common import *
import signal,subprocess

def main():
    target=HERE/'CPU_PLACEMENT_HANDOFF.receipt.json';assert not target.exists()
    proof=read(HERE/'CHECKPOINT_COMPATIBILITY.json');assert proof['passed']
    for p,h in proof['implementation_sources'].items():assert sha(p)==h,p
    cpuinfo=Path('/proc/cpuinfo').read_text();flags={}
    for block in cpuinfo.split('\n\n'):
        fields=dict(line.split(':',1)for line in block.splitlines()if':'in line)
        if 'processor\t'in fields:flags[int(fields['processor\t'].strip())]=fields.get('flags\t\t','').strip()
    speeds={}
    for p in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
        cpu=int(p.name[3:]);f=p/'cpufreq/cpuinfo_max_freq'
        if f.exists():speeds[cpu]=int(f.read_text())
    fastest=max(speeds.values());pcores=[cpu for cpu,speed in speeds.items()if speed>=(fastest*.98)]
    cases={c['rollout_uid']:c for c in read(HERE/'main_MISSING_CASES.json')};plans=[];owners=[]
    for p in(HERE/'QUEUES/main/uid_claims').glob('*/OWNER.json'):
        owner=read(p);rid=p.parent.name
        if(HERE/'main/raw'/f'{rid}.json').exists():continue
        pid=owner['pid'];proc=Path(f'/proc/{pid}')
        if not proc.exists():continue
        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert'cross_scene_pipeline_v2/rollout_pool_v2.py main'in cmd
        affinity=sorted(os.sched_getaffinity(pid))
        if max(speeds[cpu]for cpu in affinity)>=fastest*.98:continue
        c=cases[rid];assert c.get('cross_scene_new')and c['controller_payload']['nominal']['type_code']=='prioritized_astar'
        assert all(flags[cpu]==flags[pcores[0]]for cpu in affinity), 'CPU instruction flags differ'
        saved=sorted((HERE/'operational_checkpoints'/rid).glob('*.receipt.json'),key=lambda q:q.stat().st_mtime);assert saved
        snap=read(saved[-1]);assert snap['scientific_label']is None and sha(snap['path'])==snap['sha256']
        owners.append(dict(owner=owner,rollout_uid=rid,scene=c['state']['scene'],state_index=c['state']['index'],old_affinity=affinity,
            old_max_frequency_khz=[speeds[cpu]for cpu in affinity],last_receipt_before_signal=str(saved[-1]),last_step_before_signal=snap['next_step']))
        plans.append({k:c[k]for k in('state_uid','eta_uid','controller_uid','seed_key')})
    assert owners
    path=HERE/'CPU_PLACEMENT_PLANNED_ROLLOUTS.json';write(path,dict(requests=plans));pf=planner.preflight(path);write(HERE/'CPU_PLACEMENT_OFFICIAL_PREFLIGHT.json',pf)
    assert pf['summary']['genuinely_missing']==len(owners)and pf['summary']['ambiguous']==0
    receipt=dict(utc=utc(),script_sha256=sha(__file__),scope='Operational CPU placement only. Existing handler saves exact snapshots and exits without scientific labels; unchanged pool loads them after old PID exits. No CPU affinity bypass and no source edits.',
        performance_core_candidates=pcores,max_cpu_frequency_khz=speeds,matching_instruction_flags_verified=True,checkpoint_compatibility_sha256=sha(HERE/'CHECKPOINT_COMPATIBILITY.json'),
        source_files=proof['implementation_sources'],owners=owners,signals_sent=False)
    write(target,receipt)
    for row in owners:os.kill(row['owner']['pid'],signal.SIGUSR1)
    receipt['signals_sent']=True;write(target,receipt)
    print('Checkpointed E-core allocations for normal Slurm reassignment:',[(r['owner']['job'],r['state_index'],r['old_affinity'])for r in owners],flush=True)

if __name__=='__main__':main()
