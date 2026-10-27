"""Eligibility rollouts for the second test batch (same rule as step3): states 0-7, 49 menu entries, 3 original controllers."""
import json,run_any
from run import KEYS
if __name__=='__main__':
 S=[s['name'] for s in json.load(open('scenes_test2.json'))]
 run_any.go([('t2_gate.jsonl',v,ci,si,'menu:'+K,None) for v in S for ci in range(3) for si in range(8) for K in KEYS],12)
