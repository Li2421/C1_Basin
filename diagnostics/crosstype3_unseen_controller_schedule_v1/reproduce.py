"""Independent sealed preflight and pre-hashes for32 mandatory engineering replays."""
from common import *
C1=D/'crosstype1c_two_eta_schedule_v1';C2=D/'crosstype2_mixed_zoo_selector_v1'
def design():
    prereg();assert not (HERE/'REPRODUCTION_DESIGN.receipt.json').exists();rows=[];base=read(C1/'BASE_PROTOCOL.json')
    counts={'Flow':3,'gaussian':2,'deterministic':2,'mixture':2,'diffusion':2,'chunking':2,'recurrent':1,'cvae':1,'ring_ippo':1}
    for kind,count in counts.items():
        phase='pilot' if kind in ('Flow','diffusion','ring_ippo') else 'remaining';cases=read(C1/f'{phase}_CASES.json');names=[n for n in base['controller_order'] if base['types'][n]==kind]
        if kind=='Flow':names=['base_u_v10_local_macflow',*[n for n in names if n!='base_u_v10_local_macflow']]
        for j,name in enumerate(names[:count]):
            c=next(c for c in cases if c['base_controller_name']==name and c['phase_state_index']==0 and c['training_slot']==8)
            path=C1/phase/'raw'/f"{c['rollout_uid']}.json";ref=read(path);assert sha(ref['trace_path'])==ref['trace_sha256'];rows.append(dict(source='c1c',type=kind,case=c,reference=str(path),reference_sha256=sha(path),trace_sha256=ref['trace_sha256'],protocol=str(C1/f'{phase}_protocol.json')))
    cases=read(C2/'new_evaluation_CASES.json')
    for kind,count in [('centralized_mppi',6),('distributed_cross_entropy',5),('prioritized_astar',5)]:
        names=[n for n in read(C2/'new_evaluation_protocol.json')['controller_order'] if read(C2/'new_evaluation_protocol.json')['types'][n]==kind]
        for j,name in enumerate(names[:count]):
            index=(11,12,0,1,11,12)[j];c=next(c for c in cases if c['base_controller_name']==name and c['seed']==0 and c['phase_state_index']==0 and c['candidate_index']==index);path=C2/'new_evaluation/raw'/f"{c['rollout_uid']}.json";ref=read(path);assert sha(ref['trace_path'])==ref['trace_sha256'];rows.append(dict(source='c2',type=kind,case=c,reference=str(path),reference_sha256=sha(path),trace_sha256=ref['trace_sha256'],protocol=str(C2/'new_evaluation_protocol.json')))
    assert len(rows)==32
    write(HERE/'REPRODUCTION_CASES.json',rows);plan=HERE/'reproduction_PLANNED_ROLLOUTS.json';write(plan,dict(requests=[{k:r['case'][k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for r in rows]));pre=planner.preflight(plan);write(HERE/'reproduction_CACHE_PREFLIGHT.json',pre)
    with rollout_db.connect(True) as db:assert all(db.execute('SELECT1'.replace('SELECT1','SELECT 1 FROM rollout WHERE rollout_uid=?'),(r['case']['rollout_uid'],)).fetchone() is not None for r in rows)
    seal('REPRODUCTION_DESIGN.receipt.json',[HERE/'REPRODUCTION_CASES.json',plan,HERE/'reproduction_CACHE_PREFLIGHT.json',HERE/'reproduce.py',HERE/'BOOTSTRAP_SOURCES.json'],before_replay=True,old_trace_hash_verified_before_replay=True,verification_only_no_new_task_labels=True)
    print('Sealed32 mandatory exact cached replay identities with pre-verified original traces',flush=True)
def run(worker):
    from mixed_execution import execute
    rows=read(HERE/'REPRODUCTION_CASES.json')
    for i in range(worker,len(rows),16):
        path=HERE/f'reproduction/check_{i:02}.json';assert not path.exists();r=rows[i];assert sha(r['reference'])==r['reference_sha256'];ref=read(r['reference']);assert sha(ref['trace_path'])==r['trace_sha256'];out=execute(r['case'],read(r['protocol']),{},'reproduction_'+r['source']);write(HERE/f'reproduction/raw/{i:02}.json',out)
        with np.load(ref['trace_path'],allow_pickle=False) as a,np.load(out['trace_path'],allow_pickle=False) as b:x,y=a['positions'],b['positions'];positions=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
        checks=dict(first_action_bytes=np.asarray(ref['first_action'],np.float64).tobytes()==np.asarray(out['first_action'],np.float64).tobytes(),positions_bytes=positions,termination=ref['outcome']==out['outcome'],steps=ref['episode_length']==out['episode_length'],precision=not out['jax_enable_x64']);record=dict(index=i,source=r['source'],type=r['type'],controller=r['case']['base_controller_name'],checks=checks,passed=all(checks.values()));write(path,record);print(json.dumps(record),flush=True)
def collect():
    rows=[read(HERE/f'reproduction/check_{i:02}.json') for i in range(32)];types={t:all(r['passed'] for r in rows if r['type']==t) for t in sorted({r['type'] for r in rows})};r=dict(utc=utc(),passed=all(x['passed'] for x in rows),types=types,checks=rows,count=32,new_task_labels=0,official_cache_preflight_before_replay=True)
    write(HERE/'EXECUTION_REPRODUCTION.json',r);seal('EXECUTION_REPRODUCTION.receipt.json',[HERE/'EXECUTION_REPRODUCTION.json',HERE/'REPRODUCTION_DESIGN.receipt.json',HERE/'reproduce.py'],no_db_writes=True);assert any(types.values()),'All reproduced paths failed'
    print('Execution reproduction:',r['passed'],types,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='run':run(int(sys.argv[2]))
    else:globals()[sys.argv[1]]()
