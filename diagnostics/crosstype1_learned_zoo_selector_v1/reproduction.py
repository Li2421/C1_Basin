"""Exactly 16 round38 + 16 round37 historical replay checks, no relabeling."""
from common import *
from episode import execute

def design():
    prereg()
    requests = []
    rows = []
    # Balanced across required learned types, without selecting by outcome.
    old = D / 'blackbox_families_v1'
    cases = read(old / 'main_CASES.json')
    wanted = [x for x in cases if x['family'] in ('deterministic','mixture','diffusion')
              and x['arm'] in ('F10','F13','S500','Sprog')]
    chosen = []
    for family, count in [('deterministic',6), ('mixture',5), ('diffusion',5)]:
        available = [x for x in wanted if x['family'] == family]
        chosen.extend(available[:count])
    for c in chosen:
        raw = old / 'main/raw' / (c['rollout_uid'] + '.json')
        rows.append(dict(group='round38', case=c, protocol=str(old/'main_protocol.json'),
                         reference=str(raw), reference_sha256=sha(raw)))
    old = D / 'blackbox_addendum_v1'
    cases = read(old / 'right_hand_evaluation_CASES.json')
    for kind in ('Flow','G'):
        available = [x for x in cases if x['kind']==kind and x['r']==.9
                     and x['base_controller_name']!='base_u_v10_local_macflow'][:8]
        assert len(available)==8
        for c in available:
            raw = old / 'right_hand_evaluation/raw' / (c['rollout_uid'] + '.json')
            rows.append(dict(group='round37', case=c, protocol=str(old/'right_hand_evaluation_protocol.json'),
                             reference=str(raw), reference_sha256=sha(raw)))
    assert len(rows)==32
    for row in rows:
        oldraw = read(row['reference'])
        assert sha(oldraw['trace_path']) == oldraw['trace_sha256']
        requests.append({k:row['case'][k] for k in ('state_uid','eta_uid','controller_uid','seed_key')})
    write(HERE/'REPRODUCTION_CASES.json', rows)
    path=HERE/'reproduction_PLANNED_ROLLOUTS.json'
    write(path, dict(requests=requests))
    pf=planner.preflight(path)
    write(HERE/'reproduction_CACHE_PREFLIGHT.json',pf)
    seal('REPRODUCTION_DESIGN.receipt.json', [HERE/'REPRODUCTION_CASES.json', path,
         HERE/'reproduction_CACHE_PREFLIGHT.json', HERE/'episode.py', HERE/'reproduction.py'],
         exception='32 mandatory reproduction checks only; no new labels or journals',
         historical_references_present=True)

def run(index):
    rows=read(HERE/'REPRODUCTION_CASES.json')
    checks=[]
    for number,row in enumerate(rows):
        if number % 2 != index:
            continue
        target=HERE/'reproduction/checks'/f'{number:02}.json'
        if target.exists():
            checks.append(read(target));continue
        p=read(row['protocol']);c=dict(row['case'])
        assert sha(row['reference']) == row['reference_sha256']
        reference=read(row['reference'])
        if row['group']=='round37':
            cp=p['profiles'][c['profile_key']]
            p['profiles']={c['base_controller_name']:cp}
            c['placement']='FF' if c['kind']=='Flow' else 'TT'
        result=execute(c,p,{},'reproduction')
        write(HERE/'reproduction/raw'/f'{number:02}.json',result)
        assert sha(reference['trace_path']) == reference['trace_sha256']
        with np.load(reference['trace_path']) as a, np.load(result['trace_path']) as b:
            aa,bb=a['positions'],b['positions']
            positions=aa.shape==bb.shape and aa.dtype==bb.dtype and aa.tobytes()==bb.tobytes()
        first=(reference['first_action'] is None and result['first_action'] is None) or (
               reference['first_action'] is not None and result['first_action'] is not None
               and np.asarray(reference['first_action'],np.float64).tobytes()==np.asarray(result['first_action'],np.float64).tobytes())
        check=dict(group=row['group'], controller=c['base_controller_name'], reference=row['reference'],
                   first_action_bytes=first, positions_bytes=positions,
                   termination=reference['outcome']==result['outcome'],
                   steps=reference['episode_length']==result['episode_length'],
                   jax_enable_x64=result['jax_enable_x64'])
        check['passed']=all(check[k] for k in ('first_action_bytes','positions_bytes','termination','steps'))
        write(target,check);checks.append(check)
        print(json.dumps(check),flush=True)

def collect():
    paths=sorted((HERE/'reproduction/checks').glob('*.json'))
    assert len(paths)==32
    checks=[read(p) for p in paths]
    families={f:all(r['passed'] for r in checks if r['controller'].startswith(f))
              for f in ('deterministic','mixture','diffusion','G','base_u')}
    write(HERE/'REPRODUCTION_GATE.json',dict(utc=utc(),checks=checks,passed=all(families.values()),
          type_passed=families, count=len(checks), new_task_labels=0))

if __name__=='__main__':
    if sys.argv[1]=='run':run(int(sys.argv[2]))
    else:globals()[sys.argv[1]]()
