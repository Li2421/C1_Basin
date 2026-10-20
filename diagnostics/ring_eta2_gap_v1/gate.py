"""Compare all 32 Round18 replay gates byte-exactly with passed Round15 gates."""
from core import *

def main():
    features_frozen()
    assert read(R15/'CONSISTENCY_GATE.json')['passed']
    checks=[]
    for c in read(HERE/'gate_CASES.json'):
        current=HERE/'gate/raw'/f"{c['rollout_uid']}.json"
        prior=R15/'gate/raw'/f"{c['expected_round15_rollout_uid']}.json"
        r=read(current);old=read(prior)
        def same_first(a,b):
            if a['first_action'] is None or b['first_action'] is None:return a['first_action'] is None and b['first_action'] is None
            return np.asarray(a['first_action'],np.float64).tobytes()==np.asarray(b['first_action'],np.float64).tobytes()
        passed=r['outcome']==old['outcome'] and r['episode_length']==old['episode_length'] and same_first(r,old)
        checks.append(dict(case=c,passed=passed,round15_raw_sha256=sha(prior),current_raw_sha256=sha(current),
                           outcome=r['outcome'],episode_length=r['episode_length'],first_action_float64_bytes_identical=same_first(r,old)))
    passed=len(checks)==32 and all(x['passed'] for x in checks)
    write(HERE/'CONSISTENCY_GATE.json',dict(utc=utc(),passed=passed,required=32,
         criterion='Exact outcome, episode_length and first_action float64 bytes against passed Round15 gate',checks=checks))
    assert passed,'Consistency gate failed; stop main batch'
    seconds=[read(HERE/'gate/raw'/f"{c['rollout_uid']}.json")['elapsed_seconds'] for c in read(HERE/'gate_CASES.json')]
    mean=float(np.mean(seconds))
    write(HERE/'COST_GATE.json',dict(utc=utc(),mean_seconds=mean,median_seconds=float(np.median(seconds)),
         p90_seconds=float(np.quantile(seconds,.9)),planned_train_requests=len(read(HERE/'train_CASES.json')),
         planned_test_requests=len(read(HERE/'test_CASES.json')),
         estimated_CPU_hours=mean*(len(read(HERE/'train_CASES.json'))+len(read(HERE/'test_CASES.json')))/3600,
         estimated_wall_hours18=mean*(len(read(HERE/'train_CASES.json'))+len(read(HERE/'test_CASES.json')))/(18*3600)))
    print(json.dumps(dict(passed=True,cases=32,mean_seconds=mean)))

if __name__=='__main__':main()
