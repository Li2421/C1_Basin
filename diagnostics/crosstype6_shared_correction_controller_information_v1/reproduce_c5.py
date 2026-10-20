"""Required actual C5 engineering replays; never journal duplicate records."""
from common import *
OLD=D/'crosstype5_repair_gate_late_eta_v1'

def design():
    prereg();assert (HERE/'HISTORY_AUDIT.receipt.json').exists()
    rows=read(OLD/'training_CASES.json');names=['base_u_v10_local_macflow','G0','transformer_bc0','ring_ippo0','centralized_mppi0','distributed_cross_entropy0','gradient_mpc0','conflict_based_search0'];checks=[];extra=[]
    for name in names:
        for source in ('given','requery'):
            choices=[c for c in rows if c['base_controller_name']==name and c.get('late_source','given')==source]
            for c in choices:
                path=OLD/'training/raw'/f"{c['rollout_uid']}.json"
                if not path.exists():continue
                r=read(path)
                if source=='requery' and r.get('late_eta_requery') is None:continue
                trace=Path(r['trace_path']);assert trace.exists()
                checks.append(dict(case=c,reference=str(path),reference_sha256=sha(path),trace_sha256=sha(trace),protocol=str(OLD/'training_protocol.json')))
                extra.extend(dict(path=str(p),sha256=sha(p),size=p.stat().st_size) for p in (path,trace));break
            else:raise RuntimeError('No actual historical case '+name+' '+source)
    assert len(checks)==16
    write(HERE/'INPUT_ADDITIONS_02.json',dict(utc=utc(),reason='Selected historical replay actual records and traces; no success filtering',files=extra))
    write(HERE/'REPRODUCTION_CASES.json',checks)
    path=HERE/'reproduction_PLANNED_ROLLOUTS.json';write(path,dict(requests=[{k:r['case'][k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for r in checks]));write(HERE/'reproduction_CACHE_PREFLIGHT.json',planner.preflight(path))
    with rollout_db.connect(True) as db:assert all(db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(r['case']['rollout_uid'],)).fetchone() for r in checks)
    seal('REPRODUCTION_DESIGN.receipt.json',[HERE/n for n in ('reproduce_c5.py','REPRODUCTION_CASES.json','reproduction_PLANNED_ROLLOUTS.json','reproduction_CACHE_PREFLIGHT.json','PREREGISTRATION.receipt.json','HISTORY_AUDIT.receipt.json')],engineering_only=True,DB_or_journal_duplicate_writes=0)

def run(worker):
    assert not os.environ.get('XLA_FLAGS');prereg()
    from execution_v3 import execute
    row=read(HERE/'REPRODUCTION_CASES.json')[worker];ref=read(row['reference']);assert sha(row['reference'])==row['reference_sha256'] and sha(ref['trace_path'])==row['trace_sha256'];cache={}
    out=execute(row['case'],read(row['protocol']),cache,'reproduction');write(HERE/f'reproduction/raw/{worker:02}.json',out);checks={}
    with np.load(ref['trace_path'],allow_pickle=False) as a,np.load(out['trace_path'],allow_pickle=False) as b:
        for k in ('positions','velocities','actions','nominal_actions','goals'):
            checks[k]=a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes()
    checks.update(first_action_bytes=np.asarray(ref['first_action'],np.float64).tobytes()==np.asarray(out['first_action'],np.float64).tobytes(),termination=ref['outcome']==out['outcome'],steps=ref['episode_length']==out['episode_length'],precision=out['jax_enable_x64'] is False)
    record=dict(utc=utc(),controller=row['case']['base_controller_name'],type=row['case']['family'],requery=row['case'].get('late_source')=='requery',checks=checks,passed=all(checks.values()));write(HERE/f'reproduction/check_{worker:02}.json',record)
    if '_frozen_mean' in cache:cache['_frozen_mean'].close()
    print(json.dumps(record),flush=True)

def collect():
    rows=[read(HERE/f'reproduction/check_{i:02}.json') for i in range(16)];write(HERE/'EXECUTION_REPRODUCTION.json',dict(utc=utc(),passed=all(r['passed'] for r in rows),checks=rows,records=16));seal('EXECUTION_REPRODUCTION.receipt.json',[HERE/'EXECUTION_REPRODUCTION.json',HERE/'REPRODUCTION_DESIGN.receipt.json'],replays_not_journaled=True)
    assert all(r['passed'] for r in rows)

if __name__=='__main__':run(int(sys.argv[2])) if sys.argv[1]=='run' else globals()[sys.argv[1]]()
