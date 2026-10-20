"""Sixteen cache-backed replays also verify8 progress and8 zero-action records."""
from common import *
OLD=D/'crosstype3_unseen_controller_schedule_v1'
def design():
    prereg();assert (HERE/'HISTORY_AUDIT.receipt.json').exists();cases=read(OLD/'new_evaluation_CASES.json');base=read(OLD/'new_evaluation_protocol.json')
    selected=['transformer_bc0','gradient_mpc0','conflict_based_search0','centralized_mppi0','distributed_cross_entropy0','transformer_bc1','gradient_mpc1','conflict_based_search1'];rows=[]
    for arm,state in [('Sprog',0),('safety_baseline',1)]:
        for name in selected:
            c=next(c for c in cases if c['base_controller_name']==name and c['arm']==arm and c['state_index']==state and c['seed']==0);path=OLD/'new_evaluation/raw'/f"{c['rollout_uid']}.json";r=read(path);assert sha(r['trace_path'])==r['trace_sha256'];rows.append(dict(case=c,reference=str(path),reference_sha256=sha(path),trace_sha256=r['trace_sha256'],protocol=str(OLD/'new_evaluation_protocol.json'),equivalence_role='human_progress' if arm=='Sprog' else 'all_zero'))
    assert len(rows)==16 and len({r['case']['family'] for r in rows})>=5
    write(HERE/'REPRODUCTION_CASES.json',rows);path=HERE/'reproduction_PLANNED_ROLLOUTS.json';write(path,dict(requests=[{k:r['case'][k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for r in rows]));pre=planner.preflight(path);write(HERE/'reproduction_CACHE_PREFLIGHT.json',pre)
    with rollout_db.connect(True) as db:assert all(db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(r['case']['rollout_uid'],)).fetchone() is not None for r in rows)
    seal('REPRODUCTION_DESIGN.receipt.json',[HERE/'REPRODUCTION_CASES.json',path,HERE/'reproduction_CACHE_PREFLIGHT.json',HERE/'reproduce.py',HERE/'BOOTSTRAP_SOURCES.json'],before_replay=True,original_trace_hash_verified=True,no_new_task_label_journal=True)
def run(worker):
    from new_execution import execute
    from progress import offline_switch
    r=read(HERE/'REPRODUCTION_CASES.json')[worker];ref=read(r['reference']);assert sha(r['reference'])==r['reference_sha256'] and sha(ref['trace_path'])==r['trace_sha256'];out=execute(r['case'],read(r['protocol']),{},'reproduction');write(HERE/f'reproduction/raw/{worker:02}.json',out)
    with np.load(ref['trace_path'],allow_pickle=False) as a,np.load(out['trace_path'],allow_pickle=False) as b:
        x,y=a['positions'],b['positions'];positions=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes();offline=offline_switch(y,b['goals'],700,.9681620884830543) if r['equivalence_role']=='human_progress' else None
    checks=dict(first_action_bytes=np.asarray(ref['first_action'],np.float64).tobytes()==np.asarray(out['first_action'],np.float64).tobytes(),positions_bytes=positions,termination=ref['outcome']==out['outcome'],steps=ref['episode_length']==out['episode_length'],precision=not out['jax_enable_x64'])
    if r['equivalence_role']=='human_progress':checks['online_offline_switch_equal']=out['progress_switch_step']==offline
    record=dict(utc=utc(),controller=r['case']['base_controller_name'],type=r['case']['family'],role=r['equivalence_role'],checks=checks,passed=all(checks.values()));write(HERE/f'reproduction/check_{worker:02}.json',record);print(json.dumps(record),flush=True)
def collect():
    rows=[read(HERE/f'reproduction/check_{i:02}.json') for i in range(16)];types={t:all(x['passed'] for x in rows if x['type']==t) for t in sorted({r['type'] for r in rows})}
    write(HERE/'EXECUTION_REPRODUCTION.json',dict(utc=utc(),passed=all(r['passed'] for r in rows),types=types,checks=rows,progress_records=8,zero_records=8,mandatory_records=16))
    seal('EXECUTION_REPRODUCTION.receipt.json',[HERE/'EXECUTION_REPRODUCTION.json',HERE/'REPRODUCTION_DESIGN.receipt.json',HERE/'reproduce.py'],no_db_task_writes=True)
    assert any(types.values()),'All reproduced paths failed'
    print('Reproduction passed:',all(types.values()),types,flush=True)
if __name__=='__main__':run(int(sys.argv[2])) if sys.argv[1]=='run' else globals()[sys.argv[1]]()
