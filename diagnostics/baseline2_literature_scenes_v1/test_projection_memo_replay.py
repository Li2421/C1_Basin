"""Full bitwise replay of sixteen existing P2 execution records, no new states."""
from experiment import *
import numpy as np
from bottleneck_family.environment import BottleneckEnv
from bottleneck_family.scenario import Config
from diagnostics.axis4b_agent_count_v1.controllers import potential,nominal_reference,corrected_action
from shared_control.basis_families import get_basis_family
from projection_memo import MemoSafetyFilter as LiteratureSafetyFilter
import time

def main():
    source=ROOT/'diagnostics/axis4b_agent_count_v1/pilot/raw';candidates=[]
    for p in source.glob('*.json'):
        x=read(p)
        if x.get('case',{}).get('kind')=='P2' and not x.get('numerical_failure') and x['episode_length']>0:
            candidates.append((x['episode_length'],p,x))
    candidates.sort(key=lambda x:(x[0],str(x[1])))
    assert len(candidates)>=16
    rows=[];basis=get_basis_family('orthoflow3');projector=LiteratureSafetyFilter()
    for _,path,x in candidates[:16]:
        start=time.monotonic();assert sha(x['trace_path'])==x['trace_sha256'];z=np.load(x['trace_path']);c=x['case'];env=BottleneckEnv(Config(**c['evaluation_config']));env.reset(np.asarray(x['initial_positions']),np.asarray(x['goals']))
        pp=[env.positions.copy()];aa=[];error=None
        try:
            for t in range(c['evaluation_config']['max_steps']):
                ref=nominal_reference(potential(env),env)
                if c['arm']=='nominal_only':a=ref
                else:a,_,_=corrected_action(env,ref,c['eta_a'] if t<c['switch_step'] else c['eta_b'],projector,basis)
                _,done,info=env.step(a,diagnose_stalls=False);aa.append(a.copy());pp.append(env.positions.copy())
                if not np.array_equal(pp[-1],z['positions'][len(pp)-1]):
                    error=dict(first_different_step=t+1,position_max_absolute_error=float(np.max(abs(pp[-1]-z['positions'][len(pp)-1]))));break
                if done:break
        except Exception as e:error=dict(type=type(e).__name__,message=str(e))
        same_action=bool(len(aa) and aa[0].tobytes()==np.asarray(x['first_action']).tobytes())
        same_trace=bool(np.array_equal(np.asarray(pp),z['positions']))
        passed=same_action and same_trace and env.termination==x['outcome'] and len(aa)==x['episode_length'] and error is None
        rows.append(dict(source=str(path),source_sha256=sha(path),trace_sha256=x['trace_sha256'],first_action_bytes=same_action,full_positions_bitwise=same_trace,termination=env.termination,expected_termination=x['outcome'],steps=len(aa),expected_steps=x['episode_length'],passed=passed,error=error,elapsed=time.monotonic()-start))
        write(HERE/'PROJECTION_MEMO_GATE.json',dict(passed=len(rows)==16 and all(r['passed'] for r in rows),records=rows));print(len(rows),passed,len(aa),flush=True)
    seal('PROJECTION_MEMO_GATE.receipt.json',[HERE/'PROJECTION_MEMO_GATE.json',HERE/'reproduction_gate.py',HERE/'projection_adapter.py'],historical_replay_only=True)

if __name__=='__main__':main()
