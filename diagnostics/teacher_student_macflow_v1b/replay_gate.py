from core import *
from audit_utils import source_inventory
from shared_rollout_db.src.planner import preflight
from episode_frozen import episode
from pipeline import make_runtime
import jax
jax.config.update('jax_enable_x64',False)

def design():
    source=D/'decentralized_pipeline_v1';p=read(source/'main_protocol.json');names=p['controller_order'][:3]+[next(n for n in p['controller_order'] if 'local' in n)]
    assert len(set(names))==4
    rows=[c for c in read(source/'main_CASES.json') if c['base_controller_name'] in names and c['pipeline']=='local' and c['arm'] in ['F10','S500'] and c['state_index'] in [0,1] and c['seed']==0];assert len(rows)==16
    write(HERE/'gate_protocol.json',p);write(HERE/'REPLAY_CASES.json',rows)
    manifest=HERE/'REPLAY_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in rows]));pf=preflight(manifest);assert pf['summary']['exact_reusable']==16 and pf['summary']['genuinely_missing']==0
    write(HERE/'REPLAY_CACHE_PREFLIGHT.json',pf);seal('REPLAY_PREFLIGHT.receipt.json',[manifest,HERE/'REPLAY_CACHE_PREFLIGHT.json',HERE/'REPLAY_CASES.json',HERE/'pipeline.py',HERE/'episode_frozen.py',HERE/'core.py'],before_submission=True,purpose='mandated consistency replay, no new DB labels')
def run():
    assert (HERE/'REPLAY_PREFLIGHT.receipt.json').exists();p=read(HERE/'gate_protocol.json');cache={};results=[]
    for c in read(HERE/'REPLAY_CASES.json'):
        r=episode(c,p,cache,'gate');ref=read(D/'decentralized_pipeline_v1/main/raw'/f"{c['rollout_uid']}.json");a=np.load(r['trace_path'])['positions'];b=np.load(ref['trace_path'])['positions']
        flags=dict(first_action_bytes=np.asarray(r['first_action'],dtype=np.float64).tobytes()==np.asarray(ref['first_action'],dtype=np.float64).tobytes(),outcome=r['outcome']==ref['outcome'],steps=r['episode_length']==ref['episode_length'],positions=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes())
        results.append(dict(case=c,checks=flags,passed=all(flags.values()),replay=r,reference_sha256=sha(D/'decentralized_pipeline_v1/main/raw'/f"{c['rollout_uid']}.json")))
        write(HERE/'ROUND28_REPLAY_GATE.json',dict(utc=utc(),passed=all(x['passed'] for x in results),completed=len(results),required=16,results=results));assert all(flags.values()),'Round28 replay gate failed'
        print('replay PASS',len(results),c['base_controller_name'],c['arm'],flush=True)
    source_inventory('imports_replay.json');seal('ROUND28_REPLAY_GATE.receipt.json',[HERE/'ROUND28_REPLAY_GATE.json',HERE/'pipeline.py',HERE/'episode_frozen.py',HERE/'core.py'],before_student_rollouts=True)
if __name__=='__main__':globals()[sys.argv[1]]()
