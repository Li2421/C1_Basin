from core import *

def main():
    prereg();rows=read(HERE/'gate_CASES.json');oldcases=read(R4/'CASES.json');lookup={(c['controller_name'],c['state_uid'],c['eta_index'],c['seed']):c for c in oldcases if c['split']=='loco_test'};checks=[]
    for c in rows:
        r=read(HERE/'gate/raw'/f"{c['rollout_uid']}.json");expected=lookup[c['base_controller_name'],c['state_uid'],c['expected_eta_index'],c['seed']];old=read(R4/'raw'/f"{expected['rollout_uid']}.json")
        first_same=r['first_action'] is None and old['first_action'] is None
        if r['first_action'] is not None and old['first_action'] is not None:first_same=np.asarray(r['first_action'],np.float64).tobytes()==np.asarray(old['first_action'],np.float64).tobytes()
        same=r['outcome']==old['outcome'] and r['episode_length']==old['episode_length'] and first_same
        checks.append(dict(case=c,old_rollout_uid=expected['rollout_uid'],old_raw_path=str(R4/'raw'/f"{expected['rollout_uid']}.json"),new_outcome=r['outcome'],old_outcome=old['outcome'],new_episode_length=r['episode_length'],old_episode_length=old['episode_length'],first_action_float64_bytes_identical=first_same,summary_identical=r['summary']==old['summary'],error_identical=r['error']==old['error'],passed=same,elapsed_seconds=r['elapsed_seconds'],trace_sha256=r['trace_sha256']))
    passed=len(checks)==32 and all(x['passed'] for x in checks);write(HERE/'CONSISTENCY_GATE.json',dict(utc=utc(),passed=passed,status='PASS' if passed else 'FAIL STOP MAIN',checks=checks,required_cases=32,observed_cases=len(checks),criterion='exact outcome, episode_length, first_action float64 bytes; no tolerance',main_allowed=passed));print(json.dumps(dict(passed=passed,failed=[x for x in checks if not x['passed']])));assert passed,'Stop and report: mandatory consistency gate failed'
    secs=np.array([r['elapsed_seconds'] for r in checks]);write(HERE/'COST_GATE.json',dict(utc=utc(),new_main_requests=10240,mean_gate_episode_seconds=float(secs.mean()),median_gate_seconds=float(np.median(secs)),p90_gate_seconds=float(np.quantile(secs,.9)),CPU_hours_estimate=float(secs.mean()*10240/3600),wall_hours_at13CPU=float(secs.mean()*10240/(13*3600)),caveat='Pure-eta validation timing; switched trajectories and worker tails may differ; no budget arms removed'))
if __name__=='__main__':main()
