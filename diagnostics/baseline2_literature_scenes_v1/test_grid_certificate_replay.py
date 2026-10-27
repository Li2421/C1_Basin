"""Sixteen complete existing planner episodes must match bitwise."""
from experiment import *
from phase_reports import records
from flow_runtime_certified import run
import numpy as np
import time

def main():
    candidates=[r for r in records('capability_lit_circle_five') if not r['numerical_failure'] and not r.get('deterministic_source')]
    candidates.sort(key=lambda x:(x['episode_length'],x['rollout_uid']));bytype={}
    for r in candidates:
        t=r['case']['controller_payload']['nominal']['type_code']
        if len(bytype.setdefault(t,[]))<2:bytype[t].append(r)
    candidates=[r for t in sorted(bytype) for r in bytype[t]];assert len(candidates)==16,len(candidates)
    rows=[]
    for old in candidates[:16]:
        assert sha(old['trace_path'])==old['trace_sha256'];start=time.monotonic();new,arrays=run(old['case'])
        with np.load(old['trace_path']) as z:
            first=bool(np.array_equal(arrays['actions'][:1],z['actions'][:1]));full=bool(np.array_equal(arrays['positions'],z['positions']))
        passed=first and full and new['outcome']==old['outcome'] and new['episode_length']==old['episode_length']
        rows.append(dict(rollout_uid=old['rollout_uid'],trace_sha256=old['trace_sha256'],first_action_bits=first,full_positions_bitwise=full,termination_equal=new['outcome']==old['outcome'],steps_equal=new['episode_length']==old['episode_length'],passed=passed,seconds=time.monotonic()-start))
        write(HERE/'GRID_CERTIFICATE_REPLAY.json',dict(passed=len(rows)==16 and all(r['passed'] for r in rows),records=rows));print(len(rows),passed,round(rows[-1]['seconds'],2),flush=True)
    assert all(r['passed'] for r in rows)
    files=[HERE/n for n in ('MEMO_BACKEND_BUILD.json','GRID_CERTIFICATE_REPLAY.json','memo_planning.py','flow_runtime_memo.py','astar_kernel_memo.cpp','cbs_kernel_memo.cpp','astar_kernel_memo.so','cbs_kernel_memo.so')]
    seal('GRID_CERTIFICATE_REPLAY_ONLY.receipt.json',files,exact_functional_memoization=True,canonical_semantic_cache_identity_preserved=True,new_research_rollouts=0)
if __name__=='__main__':main()
