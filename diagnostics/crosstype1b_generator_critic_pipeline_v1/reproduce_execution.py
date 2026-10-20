"""Sixteen mandatory bitwise replays covering all nine learned types."""
from common import *

OLD=D/'crosstype1_learned_zoo_selector_v1'


def design():
    import csv
    prereg()
    with (OLD/'ZOO_MANIFEST.csv').open() as f:types={r['controller']:r['type_code'] for r in csv.DictReader(f)}
    stages=('old_evaluation','new_evaluation','locked_evaluation');available=[]
    for stage in stages:
        for c in read(OLD/f'{stage}_CASES.json'):
            if c['seed']!=0:continue
            path=OLD/stage/'raw'/f"{c['rollout_uid']}.json"
            if path.exists():available.append((c,path,OLD/f'{stage}_protocol.json'))
    wanted={'Flow':2,'gaussian':2,'deterministic':2,'mixture':2,'diffusion':2,'chunking':2,'recurrent':2,'cvae':1,'ring_ippo':1}
    rows=[]
    preferred=('Sprog','S500','F13','S100','safety_baseline','F10','nominal_only','S300','RHR')
    for kind,count in wanted.items():
        names=sorted({c['base_controller_name'] for c,_,_ in available if types[c['base_controller_name']]==kind})
        if kind=='Flow':names=['base_u_v10_local_macflow',next(n for n in names if n!='base_u_v10_local_macflow')]
        for name in names[:count]:
            pool=[r for r in available if r[0]['base_controller_name']==name]
            c,path,protocol=min(pool,key=lambda r:(preferred.index(r[0]['arm']),r[0]['state_index']))
            ref=read(path);assert sha(ref['trace_path'])==ref['trace_sha256']
            rows.append(dict(type=kind,case=c,reference=str(path),reference_sha256=sha(path),protocol=str(protocol)))
    assert len(rows)==16
    write(HERE/'REPRODUCTION_CASES.json',rows)
    path=HERE/'reproduction_PLANNED_ROLLOUTS.json'
    write(path,dict(requests=[{k:r['case'][k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for r in rows]))
    write(HERE/'reproduction_CACHE_PREFLIGHT.json',planner.preflight(path))
    seal('REPRODUCTION_DESIGN.receipt.json',[HERE/'REPRODUCTION_CASES.json',path,HERE/'reproduction_CACHE_PREFLIGHT.json'],
        mandatory_replay_only=True,new_task_labels=0,selection='First available stored records by declared arm order, original state index; cover distinct controllers and native Flow, no outcome selection')


def run():
    from rollout_runtime import execute
    checks=[]
    for i,r in enumerate(read(HERE/'REPRODUCTION_CASES.json')):
        assert sha(r['reference'])==r['reference_sha256']
        ref=read(r['reference']);p=read(r['protocol']);c=r['case']
        result=execute(c,p,{},'reproduction')
        write(HERE/'reproduction/raw'/f'{i:02}.json',result)
        with np.load(ref['trace_path'],allow_pickle=False) as a,np.load(result['trace_path'],allow_pickle=False) as b:
            x,y=a['positions'],b['positions'];positions=x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
        first=(ref['first_action'] is None and result['first_action'] is None) or (
            ref['first_action'] is not None and result['first_action'] is not None and
            np.asarray(ref['first_action'],np.float64).tobytes()==np.asarray(result['first_action'],np.float64).tobytes())
        check=dict(type=r['type'],controller=c['base_controller_name'],first_action_bytes=first,positions_bytes=positions,
            termination=ref['outcome']==result['outcome'],steps=ref['episode_length']==result['episode_length'],x64=result['jax_enable_x64'])
        check['passed']=all(check[k] for k in ('first_action_bytes','positions_bytes','termination','steps'))
        write(HERE/'reproduction/checks'/f'{i:02}.json',check);checks.append(check);print(json.dumps(check),flush=True)
    result=dict(utc=utc(),checks=checks,passed=all(c['passed'] for c in checks),count=16,types=9,new_task_labels=0)
    write(HERE/'EXECUTION_REPRODUCTION.json',result)
    seal('EXECUTION_REPRODUCTION.receipt.json',[HERE/'EXECUTION_REPRODUCTION.json',HERE/'reproduce_execution.py',HERE/'rollout_runtime.py'],passed=result['passed'])


if __name__=='__main__':globals()[sys.argv[1]]()
