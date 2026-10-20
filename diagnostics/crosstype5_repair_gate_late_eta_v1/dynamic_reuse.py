"""Reuse newly completed actual records; never use an inferred record as source."""
from common import *
from actions import fixed_source_uid
from early_reuse import proof,validated

def infer_actual_file(c,base,stage):
    if c['family'] not in validated() or np.array_equal(c['eta_a'],c['eta_b']):return None
    rid=fixed_source_uid(base,c['base_controller_name'],c['state_uid'],c['eta_a'],c['seed']);path=HERE/stage/'raw'/f'{rid}.json'
    if rid==c['rollout_uid'] or not path.exists():return None
    r=read(path);assert r['rollout_uid']==rid and r['case']['state_uid']==c['state_uid'];assert np.array_equal(r['case']['eta_a'],r['case']['eta_b']);assert sha(r['trace_path'])==r['trace_sha256']
    source=dict(record=r,trace_path=r['trace_path'],trace_sha256=r['trace_sha256'],record_source_path=str(path),source_rollout_uid=rid);p=proof(c,source)
    if p is None:return None
    return dict(target_rollout_uid=c['rollout_uid'],source_rollout_uid=rid,kind='提前结束等价复用',source_record_file=str(path),source_trace_path=r['trace_path'],source_trace_sha256=r['trace_sha256'],proof=p,outcome={k:r[k] for k in ('success','collision','timeout','deadlock','numerical_failure','outcome','episode_length','physical_steps')},written_to_shared_DB=False,dynamic_actual_source=True)

def aliases(stage):
    result=read(HERE/f'{stage}_LOCAL_EQUIVALENCE.json')
    folder=HERE/stage/'local_equivalence'
    if folder.exists():
        for p in folder.glob('*.json'):
            r=read(p);rid=r['target_rollout_uid'];assert rid not in result;result[rid]=r
    return result
