"""Conditional old-runtime versus v2 per-step byte trace comparison."""
from __future__ import annotations
import datetime, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
from adapter import historical_state,intervention_runtime
from new_benchmark_common import basin_dataset_v1 as bd
old_path=ROOT/'diagnostics/orthoflow3_controller_intervention_generalization_v1/intervention.py'
spec=importlib.util.spec_from_file_location('historical_intervention_local',old_path)
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrhex(a):return np.asarray(a,dtype=np.float64).tobytes().hex()

class IsolatedHistoricalRuntime(bd.TrainingRuntime):
    def _register_training_identities(self):pass

def instrument(rt):
    original=rt.make_env;trace=[]
    def make():
        env=original();step=env.step
        def wrapped(action):
            before=dict(position=arrhex(env.positions),velocity=arrhex(env.velocities),action=arrhex(action))
            out=step(action)
            before.update(position_after=arrhex(env.positions),velocity_after=arrhex(env.velocities))
            trace.append(before)
            return out
        env.step=wrapped
        return env
    rt.make_env=make
    return trace

def main():
    assert read(HERE/'G1A_REPORT.json')['status']=='PASS'
    s=read(HERE/'G1A_SELECTION.json');records={r['rollout_uid']:r for r in s['records']}
    states=read(ROOT/'diagnostics/orthoflow3_loso_partial_count_v1/states.json')
    original_class=bd.TrainingRuntime
    results=[]
    for uid in s['g1b_rollout_uids']:
        r=records[uid];index=r['original']['initial_state_index'];state=historical_state(states[index],index)
        if r['experiment_uid']=='exp_orthoflow3_controller_intervention_balanced_v1':
            out=ROOT/'diagnostics/orthoflow3_controller_intervention_generalization_v1/balanced_expansion'
        elif r['experiment_uid']=='exp_orthoflow3_controller_intervention_second_variant_v1':
            out=ROOT/'diagnostics/orthoflow3_controller_intervention_generalization_v1/second_variant'
        else:raise AssertionError(r['experiment_uid'])
        protocol=read(out/'protocol.json')
        profile=protocol['profiles']['four_way_intersection']
        assert sha(profile['alternate_path'])==r['checkpoint_sha256']
        old.OUT=out
        try:
            bd.TrainingRuntime=IsolatedHistoricalRuntime
            historic,by,reset_old=old._runtime('four_way_intersection',profile)
        finally:
            bd.TrainingRuntime=original_class
        assert uid in records and r['state_uid'] in by
        trace_old=instrument(historic)
        reset_old()
        result_old=historic.rollout(by[r['state_uid']],np.asarray(r['original']['eta']),r['original']['future_index'],'orthoflow3')
        modern,reset_new=intervention_runtime(Path(r['checkpoint_path']))
        trace_new=instrument(modern)
        reset_new()
        result_new=modern.rollout(state,np.asarray(r['original']['eta']),r['original']['future_index'],'orthoflow3')
        comparisons=[]
        for i,(a,b) in enumerate(zip(trace_old,trace_new)):
            for key in ('position','velocity','action','position_after','velocity_after'):
                if a[key]!=b[key]:comparisons.append(dict(step=i,field=key))
        same_length=len(trace_old)==len(trace_new)
        same_first=bool(trace_old and trace_new and trace_old[0]['action']==trace_new[0]['action'])
        passed=same_length and same_first and not comparisons and result_old['outcome']==result_new['outcome'] and result_old['episode_length']==result_new['episode_length']
        item=dict(rollout_uid=uid,run=r['run'],outcome_stratum=r['outcome_stratum'],old_runtime_path=str(Path(old.__file__)),
                  old_runtime_sha256=sha(old.__file__),old_steps=len(trace_old),new_steps=len(trace_new),
                  same_first_action_bytes=same_first,same_length=same_length,mismatches=comparisons,
                  old_outcome=result_old['outcome'],new_outcome=result_new['outcome'],passed=passed,
                  old_trace_sha256=hashlib.sha256(json.dumps(trace_old,sort_keys=True).encode()).hexdigest(),
                  new_trace_sha256=hashlib.sha256(json.dumps(trace_new,sort_keys=True).encode()).hexdigest())
        results.append(item)
        print(json.dumps({'gate':'G1b','uid':uid,'passed':passed,'steps':len(trace_old)}),flush=True)
    status='PASS' if all(x['passed'] for x in results) and len(results)==4 else 'MISMATCH'
    report=dict(gate='G1b',status=status,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                selection_sha256=sha(HERE/'G1A_SELECTION.json'),adapter_sha256=sha(HERE/'adapter.py'),results=results)
    (HERE/'G1B_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'gate':'G1b','status':status,'passed':sum(x['passed'] for x in results),'required':4}),flush=True)
if __name__=='__main__':main()
