"""Round36 native chain replay: preserve controller, keys and original physical state."""
from common import *
from registry import register, postflight
from shared_rollout_db.src.rollout_db import uid,canonical
import subprocess
STAGE='reproduction_v2'

def prepare():
    old=D/'axis4b_agent_count_v1';p=read(old/'pilot_protocol.json')
    cases=[c for c in read(old/'pilot_CASES.json') if c['kind']=='P2' and c['N']==2 and c['arm']=='safety_baseline'][:16]
    assert len(cases)==16
    rows=[]
    for original in cases:
        c=dict(original);c['reference_raw']=str(old/'pilot/raw'/f"{original['rollout_uid']}.json")
        ref=read(c['reference_raw']);assert sha(ref['trace_path'])==ref['trace_sha256']
        c['seed']=3900
        c['seed_key']=canonical(dict(future_index=c['seed']))
        c['rollout_uid']=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))
        rows.append(c)
    write(HERE/f'{STAGE}_protocol.json',p)
    register(STAGE,rows,[])
    seal('REPRODUCTION_IMPLEMENTATION_V2.receipt.json',[HERE/f for f in ['core.py','common.py','runtime.py','controllers.py','reproduction.py',f'{STAGE}_protocol.json']],reason='Prompt explicitly requires sixteen bitwise records; P2 is deterministic and ignores future keys; genuinely missing future index3900 uses official seed semantics')

def execute():
    from runtime import execute as run
    prereg();p=read(HERE/f'{STAGE}_protocol.json');cache={}
    for c in read(HERE/f'{STAGE}_MISSING_CASES.json'):
        path=HERE/STAGE/'raw'/f"{c['rollout_uid']}.json"
        if path.exists():continue
        result=run(c,p,cache,STAGE);write(path,result)
        journal=cache_writer.append_journal([result],EXP,'reproduction')
        assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals'/EXP)
    subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],check=True,stdout=(HERE/'reproduction_MERGE.log').open('w'))
    postflight(STAGE);checks=[]
    for c in read(HERE/f'{STAGE}_CASES.json'):
        old=read(c['reference_raw']);new=read(HERE/STAGE/'raw'/f"{c['rollout_uid']}.json")
        a=np.load(old['trace_path']);b=np.load(new['trace_path'])
        v=dict(first_action=np.asarray(old['first_action'],np.float64).tobytes()==np.asarray(new['first_action'],np.float64).tobytes(),
               positions=a['positions'].dtype==b['positions'].dtype and a['positions'].shape==b['positions'].shape and a['positions'].tobytes()==b['positions'].tobytes(),
               termination=old['outcome']==new['outcome'],steps=old['episode_length']==new['episode_length'])
        v.update(passed=all(v.values()),rollout_uid=c['rollout_uid'],reference_raw=c['reference_raw']);checks.append(v)
    result=dict(passed=all(v['passed'] for v in checks),records=len(checks),checks=checks)
    write(HERE/'REPRODUCTION_GATE.json',result)
    assert result['passed'], 'Historical chain blocked; no new physical evaluation'
    subprocess.run([sys.executable,'-B',str(HERE/'test_geometry.py')],check=True)

if __name__=='__main__':globals()[sys.argv[1]]()
