"""Sixteen prior actual episodes; engineering replay only, never journal."""
from common import *
OLD=D/'crosstype4_no_repair_schedule_diagnosis_v1'
def design():
    prereg();assert (HERE/'HISTORY_AUDIT.receipt.json').exists();rows=read(OLD/'B_evaluation_CASES.json');base=read(HERE/'BASE_PROTOCOL.json');names=['base_u_v10_local_macflow',next(n for n in base['controller_order'] if base['types'][n]=='Flow' and n!='base_u_v10_local_macflow'),next(n for n in base['controller_order'] if base['types'][n]=='ring_ippo_conservative'),'transformer_bc0','gradient_mpc0','conflict_based_search0','centralized_mppi0','distributed_cross_entropy0'];checks=[]
    for si,arm in [(0,'human_progress'),(1,'safety')]:
        for n in names:
            c=next(c for c in rows if c['base_controller_name']==n and c['state_index']==si and c['seed']==0 and c['arm']==arm);path=OLD/'B_evaluation/raw'/f"{c['rollout_uid']}.json";r=read(path);assert sha(r['trace_path'])==r['trace_sha256'];checks.append(dict(case=c,reference=str(path),reference_sha256=sha(path),trace_sha256=r['trace_sha256'],protocol=str(OLD/'B_evaluation_protocol.json')))
    assert len(checks)==16 and len({r['case']['family'] for r in checks})>=6
    write(HERE/'REPRODUCTION_CASES.json',checks);path=HERE/'reproduction_PLANNED_ROLLOUTS.json';write(path,dict(requests=[{k:r['case'][k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for r in checks]));pre=planner.preflight(path);write(HERE/'reproduction_CACHE_PREFLIGHT.json',pre)
    with rollout_db.connect(True) as db:assert all(db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(r['case']['rollout_uid'],)).fetchone() for r in checks)
    deps=['reproduce.py','execution.py','new_execution.py','mixed_execution.py','mixed_runtime.py','core.py','common.py','BOOTSTRAP_SOURCES.json','REPRODUCTION_CASES.json','reproduction_PLANNED_ROLLOUTS.json','reproduction_CACHE_PREFLIGHT.json'];seal('REPRODUCTION_DESIGN.receipt.json',[HERE/n for n in deps],before_replay=True,new_task_DB_writes=0)
def run(worker):
    from execution import execute
    rows=read(HERE/'REPRODUCTION_CASES.json');r=rows[worker];ref=read(r['reference']);assert sha(r['reference'])==r['reference_sha256'] and sha(ref['trace_path'])==r['trace_sha256'];out=execute(r['case'],read(r['protocol']),{},'reproduction');write(HERE/f'reproduction/raw/{worker:02}.json',out);checks={}
    with np.load(ref['trace_path'],allow_pickle=False) as a,np.load(out['trace_path'],allow_pickle=False) as b:
        for k in ('positions','velocities','actions','nominal_actions','goals'):checks[k]=a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes()
    checks.update(first_action_bytes=np.asarray(ref['first_action'],np.float64).tobytes()==np.asarray(out['first_action'],np.float64).tobytes(),termination=ref['outcome']==out['outcome'],steps=ref['episode_length']==out['episode_length'],precision=out['jax_enable_x64'] is False);record=dict(utc=utc(),controller=r['case']['base_controller_name'],type=r['case']['family'],checks=checks,passed=all(checks.values()));write(HERE/f'reproduction/check_{worker:02}.json',record);print(json.dumps(record),flush=True)
def collect():
    rows=[read(HERE/f'reproduction/check_{i:02}.json') for i in range(16)];write(HERE/'EXECUTION_REPRODUCTION.json',dict(utc=utc(),passed=all(r['passed'] for r in rows),checks=rows,records=16));seal('EXECUTION_REPRODUCTION.receipt.json',[HERE/'EXECUTION_REPRODUCTION.json',HERE/'REPRODUCTION_DESIGN.receipt.json'],replays_not_journaled=True);assert all(r['passed'] for r in rows)
if __name__=='__main__':run(int(sys.argv[2])) if sys.argv[1]=='run' else globals()[sys.argv[1]]()
