"""Reuse a completed trajectory when schedules agree at every attempted action."""
from common import *
from shared_rollout_db.src.rollout_db import canonical
import copy
from functools import lru_cache


@lru_cache(maxsize=1)
def source_index():
    index={}
    for row in read(HERE/'HISTORY_REUSE_CANDIDATES.json'):
        old=row['source']['case']
        if row['arm']=='S500':
            index.setdefault((row['controller'],old['state_uid'],old['seed']),[]).append(row['source'])
    return index


def equivalent(c,source,base_payload,source_payload):
    old=source['case']
    if c['arm'] not in ('S100','S300') or old['arm']!='S500':return False
    if any(c[k]!=old[k] for k in ('state_uid','seed_key','placement')):return False
    if np.asarray(c['eta_a'],np.float64).tobytes()!=np.asarray(old['eta_a'],np.float64).tobytes():return False
    oldbase=copy.deepcopy(source_payload);oldbase.pop('eta_schedule',None)
    if canonical(oldbase)!=canonical(base_payload):return False
    attempts=max(source['episode_length'],int(source.get('error',{}).get('step',-1))+1 if source.get('error') else 0)
    return attempts<=min(c['switch_step'],old['switch_step'])


def try_reuse(c,p):
    if c['arm'] not in ('S100','S300'):return None
    cachekey=(c['base_controller_name'],c['state_uid'],c['seed'])
    matches=source_index().get(cachekey,[])
    for record in matches:
        path=Path(record['raw_path'])
        if not path.exists():continue
        source=read(path)
        with rollout_db.connect(True) as db:
            cp=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(source['case']['controller_uid'],)).fetchone()
        if cp is None or not equivalent(c,source,p['profiles'][c['base_controller_name']]['payload'],json.loads(cp[0])):
            continue
        assert sha(source['trace_path'])==source['trace_sha256']
        result=copy.deepcopy(source)
        result.update(experiment_uid=EXP,case=c,**{k:c[k] for k in ('rollout_uid','controller_uid','state_uid','eta_uid','seed_key')},
                      physical_reuse=True,new_physical_steps=0,
                      source_raw_path=str(path),source_raw_sha256=sha(path),
                      prefix_proof=dict(schema='identical_execution_until_prior_termination_v1',
                          attempted_actions=source['episode_length'],first_possible_difference=c['switch_step'],
                          base_payload_identical=True,first_eta_bytes_identical=True,
                          proof_code_sha256=sha(__file__)))
        return result
    return None
