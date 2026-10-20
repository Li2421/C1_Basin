from core import *
from shared_rollout_db.src.planner import preflight
from registration import connect
from audit_utils import source_inventory
import signal,subprocess,ast

def main():
    assert not (HERE/'HANDOFF_CASES.json').exists()
    # Check the frozen original worker has a pre-runtime output guard outside the episode.
    tree=ast.parse((HERE/'run.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker');loop=next(n for n in f.body if isinstance(n,ast.For) and ast.unparse(n.iter).startswith('read(HERE / f'))
    assert isinstance(loop.body[0],ast.Assign) and isinstance(loop.body[1],ast.Assert);assert 'not out.exists()' in ast.unparse(loop.body[1]);assert 'episode(' not in ''.join(ast.unparse(n) for n in loop.body[:2])
    # Stop only our observer, which would otherwise misinterpret expected guard exits.
    pid=int((HERE/'CONTINUE_DELIVERY_PID').read_text());args=Path(f'/proc/{pid}/cmdline').read_bytes();assert b'continue_delivery.py' in args;os.kill(pid,signal.SIGTERM)
    lines=subprocess.check_output(['ps','-u',os.environ['USER'],'-o','pid=,args='],text=True).splitlines();workers=[]
    for line in lines:
        toks=line.split()
        if len(toks)>=7 and toks[1].endswith('/python') and 'diagnostics/teacher_student_macflow_v1d/run.py' in toks and toks[-3]=='C':workers.append((int(toks[0]),int(toks[-2])))
    assert workers and len(workers)<=7,workers
    paused=[];transfers=[];future=[]
    try:
        for pid,index in workers:os.kill(pid,signal.SIGSTOP);paused.append(pid)
        # Wait until every process is kernel-stopped; no case can advance past this snapshot.
        import time
        for pid in paused:
            for _ in range(100):
                status=Path(f'/proc/{pid}/status').read_text()
                if any(x.startswith('State:') and ('T ' in x or 'T (' in x) for x in status.splitlines()):break
                time.sleep(.01)
            else:raise AssertionError('Worker did not stop')
        allcases=read(HERE/'C_MISSING_CASES.json');(HERE/'handoff').mkdir(exist_ok=True);write(HERE/'handoff/RESERVED.json',dict(dispatch_reservation_only=True,not_a_rollout=True,not_a_DB_label=True,reason='Original worker guard stops BEFORE executing reserved future case. Replaced atomically by genuine result after helper executes it.'))
        for pid,index in workers:
            cs=allcases[index::18];first=next(k for k,c in enumerate(cs) if not (HERE/'C/raw'/f"{c['rollout_uid']}.json").exists());cut=min(first+2,len(cs));tail=cs[cut:]
            if not tail:continue
            assert not any((HERE/'C/raw'/f"{c['rollout_uid']}.json").exists() for c in tail)
            marker=HERE/'C/raw'/f"{tail[0]['rollout_uid']}.json";os.symlink(HERE/'handoff/RESERVED.json',marker)
            transfers.append(dict(pid=pid,original_worker=index,completed_prefix=first,inflight_and_retained_cases=[c['rollout_uid'] for c in cs[first:cut]],cut=cut,reserved_guard_case=tail[0]['rollout_uid'],transferred_cases=len(tail)));future.extend(tail)
        assert future and len({c['rollout_uid'] for c in future})==len(future);write(HERE/'HANDOFF_CASES.json',future);write(HERE/'HANDOFF_PLAN.json',dict(utc=utc(),transfers=transfers,source_preflight_sha256=sha(HERE/'C_PREFLIGHT.receipt.json'),original_worker_sha256=sha(HERE/'run.py'),helper_sha256=sha(HERE/'handoff_worker.py'),controls_and_physics_unchanged=True,no_inflight_cases_transferred=True,unknowns_or_completed_transferred=0))
    finally:
        for pid in paused:os.kill(pid,signal.SIGCONT)
    manifest=HERE/'HANDOFF_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in future]));pf=preflight(manifest);assert pf['summary']['ambiguous']==0
    with connect(True) as db:
        assert all(db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None for c in future)
    write(HERE/'HANDOFF_CACHE_PREFLIGHT.json',pf);seal('HANDOFF_PREFLIGHT.receipt.json',[manifest,HERE/'HANDOFF_CACHE_PREFLIGHT.json',HERE/'HANDOFF_CASES.json',HERE/'HANDOFF_PLAN.json',HERE/'handoff_worker.py',HERE/'handoff.sbatch'],before_submission=True,all_transferred_cases_previously_unstarted=True,missing=len(future));source_inventory('imports_handoff_prepare.json');print('Transferred',len(future),'future cases from',len(transfers),'workers',flush=True)
if __name__=='__main__':main()
