"""Resume orchestration with the user's immediate18CPU cap; no worker signals."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import subprocess
import signal
import time


def command(pid):
    p=Path(f'/proc/{pid}/cmdline')
    return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''


def launch(script):
    with (HERE/(Path(script).stem+'.log')).open('a') as stream:
        return subprocess.Popen([sys.executable,'-B',str(HERE/script)],cwd=ROOT,
                                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    proof='CPU_LIMIT_IMMEDIATE_18CPU_CODE_20261022.receipt.json'
    verify_receipt(proof)
    plan=read(HERE/'CPU_LIMIT_IMMEDIATE_18CPU_SOURCE_MAP_20261022.json')
    mapping=plan['mapping'];entry=read(HERE/'ACTIVE_ENTRYPOINT.json')
    old=read(HERE/'USER_RELEASE_SIX_CPUS_NOW_20261022_PLAN.json')
    assert all(str(HERE/r['script']) not in command(r['pid']) for r in old['stopped_local_managers'])
    for row in plan['live_local_processes']:
        if row['script'] in mapping and str(HERE/row['script']) in command(row['pid']):
            os.kill(row['pid'],signal.SIGTERM)
    for _ in range(40):
        if not any(row['script'] in mapping and str(HERE/row['script']) in command(row['pid']) for row in plan['live_local_processes']):break
        time.sleep(.1)
    new={k:mapping.get(v,v) if isinstance(v,str) else v for k,v in entry.items()}
    new.update(utc=utc(),total_cpu=18,total_gpu_shards=2,
               cpu_authority='Latest user: release six CPUs immediately;18 total including1 local reserve',
               resource_policy='RESOURCE_LIMIT_IMMEDIATE_18CPU_20261022.json',
               resource_request='USER_RELEASE_SIX_CPUS_NOW_20261022_PLAN.json',
               downstream_receipt=proof,optional_receipt=proof,
               resource_limit_receipt=proof)
    write(HERE/'ACTIVE_ENTRYPOINT.json',new)
    processes=[]
    for row in plan['live_local_processes']:
        if row['script'] in mapping:
            child=launch(mapping[row['script']]);processes.append(dict(pid=child.pid,script=mapping[row['script']]))
    root=launch(new['coordinator']);new['pid']=root.pid;write(HERE/'ACTIVE_ENTRYPOINT.json',new)
    processes.append(dict(pid=root.pid,script=new['coordinator']))
    for script in (new['watchdog'],new['resource_guard'],mapping['cpu_release_guard_watchdog_20261023.py']):
        child=launch(script);processes.append(dict(pid=child.pid,script=script))
        if script==new['resource_guard']:time.sleep(.5)
    write(HERE/'CPU_LIMIT_IMMEDIATE_18CPU_ACTIVATION_20261022.json',dict(utc=utc(),started=processes,active_entry=new,scientific_worker_signals=0,previous_release_selected_jobs=old['selected_jobs']))
    print('Started immediate18CPU orchestration:',processes,flush=True)


if __name__=='__main__':main()
