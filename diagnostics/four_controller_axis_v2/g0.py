"""Replay exact E8 v13 eight-case gate with the new v2 adapter."""
from __future__ import annotations
import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
from adapter import FourFlowAdapter
from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    receipt=read(HERE/'PREREGISTRATION.receipt.json');assert receipt['preregistration_sha256']==sha(HERE/'PREREGISTRATION.md')
    e8=ROOT/'diagnostics/controller_ff_factorial_repair_v1';p=read(e8/'protocol.json')
    cases=[c for c in read(e8/'cases.json') if c['scene']=='four_way_intersection' and c['controller_name']=='flow_isotropic' and c['split']=='train' and c['state_index'] in (0,1) and c['eta_index'] in (10,15) and c['seed'] in (0,1)]
    assert len(cases)==8
    src=ROOT/'diagnostics/controller_flow_endpoint_probe_v1/raw'
    ck=ROOT/'diagnostics/four_way_intersection_stage1/base_u_v13_broad_global_source_balanced_macflow/best.pkl'
    rt=FlowEndpointMatchedField(FourFlowAdapter(ck),'isotropic')
    states={s['state_uid']:s for s in p['states']};checks=[]
    for c in cases:
        oldpath=src/f"{c['rollout_uid']}.json";assert oldpath.is_file();old=read(oldpath)
        env=rt.make_env(states[c['state_uid']]);first=None;term='timeout';length=0;err=None
        for t in range(rt.config.max_steps):
            try:
                action,_,_=rt.action(env,key_for(states[c['state_uid']],c['seed'],t),np.asarray(c['eta']),'field')
                if first is None:first=np.asarray(action).copy()
                _,_,done,info=env.step(action);term=info['termination'];length=t+1
                if done:break
            except Exception as ex:
                err=f'{type(ex).__name__}: {ex}';term='numerical_failure';break
        a=np.asarray(first,np.float64).tobytes() if first is not None else None
        b=np.asarray(old['first_action'],np.float64).tobytes() if old['first_action'] is not None else None
        checks.append(dict(rollout_uid=c['rollout_uid'],source_path=str(oldpath),source_sha256=sha(oldpath),
            outcome_old=old['outcome'],outcome_new=term,length_old=old['episode_length'],length_new=length,
            first_action_bytes_match=a==b,first_action_new_be=a.hex() if a is not None else None,
            first_action_old_be=b.hex() if b is not None else None,error=err,
            passed=(term==old['outcome'] and length==old['episode_length'] and a==b)))
    passed=len(checks)==8 and all(x['passed'] for x in checks)
    out=dict(status='PASS' if passed else 'FAIL',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        gate='G0',v2_adapter_sha256=sha(HERE/'adapter.py'),preregistration_sha256=sha(HERE/'PREREGISTRATION.md'),
        checkpoint_sha256=sha(ck),checks=checks)
    (HERE/'G0_REPORT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(gate='G0',status=out['status'],passed=sum(x['passed'] for x in checks),required=8)),flush=True)
    if not passed:raise SystemExit('STOP: G0 failed')

if __name__=='__main__':main()
