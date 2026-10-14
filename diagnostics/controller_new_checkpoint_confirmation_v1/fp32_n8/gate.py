from core import *

def main():
    prereg();checks=[];r9checks={r['case']['rollout_uid']:r for r in read(R9/'CONSISTENCY_GATE.json')['checks']}
    for c in read(HERE/'gate_CASES.json'):
        r=read(HERE/'gate/raw'/f"{c['rollout_uid']}.json");old=read(R9/'gate/raw'/f"{c['expected_round9_rollout_uid']}.json");legacy_path=r9checks[c['expected_round9_rollout_uid']]['old_raw_path'];legacy=read(legacy_path)
        def same_first(a,b):
            if a['first_action'] is None or b['first_action'] is None:return a['first_action'] is None and b['first_action'] is None
            return np.asarray(a['first_action'],np.float64).tobytes()==np.asarray(b['first_action'],np.float64).tobytes()
        same=all(r['outcome']==v['outcome'] and r['episode_length']==v['episode_length'] and same_first(r,v) for v in (old,legacy));checks.append(dict(case=c,passed=same,round9_raw_sha256=sha(R9/'gate/raw'/f"{c['expected_round9_rollout_uid']}.json"),round4_raw_path=legacy_path,round4_raw_sha256=sha(legacy_path),outcome=r['outcome'],episode_length=r['episode_length'],first_action_float64_bytes_identical=same_first(r,old) and same_first(r,legacy),elapsed_seconds=r['elapsed_seconds']))
    passed=len(checks)==32 and all(r['passed'] for r in checks);write(HERE/'CONSISTENCY_GATE.json',dict(utc=utc(),passed=passed,checks=checks,required=32,criterion='Exact outcome, episode_length and first_action bytes against both round9 and round4; no tolerance'));assert passed,'Gate failed: stop main and report'
    secs=np.array([r['elapsed_seconds'] for r in checks]);write(HERE/'COST_GATE.json',dict(utc=utc(),mean_seconds=float(secs.mean()),median_seconds=float(np.median(secs)),p90_seconds=float(np.quantile(secs,.9)),main_requests=12288,estimated_CPU_hours=float(secs.mean()*12288/3600),estimated_wall_hours17=float(secs.mean()*12288/(17*3600)),uncertainty='Four historical controllers only; eight final policies may differ in cost'))
    print(json.dumps(dict(passed=passed,cases=32,mean_seconds=float(secs.mean()))))
if __name__=='__main__':main()
