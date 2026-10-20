"""Frozen 25-checkpoint structural and deterministic-forward gate."""
from __future__ import annotations
import datetime, hashlib, json, pickle, sys
from pathlib import Path
import jax
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
from adapter import FourFlowAdapter

NAMES='''base_u_plus_w_v5_outer_buffer_dense_macflow
base_u_plus_w_v5_outer_buffer_dense_macflow_early
base_u_plus_w_v5_outer_buffer_dense_macflow_seed3111_superseded
base_u_plus_w_v5_outer_buffer_macflow
base_u_plus_w_v6_policy_uniform_macflow
base_u_v10_world_timeout_goal_recovery_source_balanced_macflow
base_u_v10_world_timeout_goal_recovery_source_balanced_wide_macflow
base_u_v11_world_timeout_agent_recovery_source_balanced_macflow
base_u_v12_world_timeout_goal_recovery_canonical_source_balanced_macflow
base_u_v13_broad_global_source_balanced_macflow
base_u_v5_outer_buffer_macflow
base_u_v9_world_goal_recovery_macflow
base_u_v9_world_goal_recovery_source_balanced_macflow'''.splitlines()
NORMALIZERS={'obs_mean','obs_scale','act_mean','act_scale'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def payload(p):
    with Path(p).open('rb') as f:return pickle.load(f)
def fingerprint(x):
    from flax.core import unfreeze
    from flax import traverse_util
    tree=traverse_util.flatten_dict(unfreeze(x['agent']['network']['params']))
    h=hashlib.sha256()
    for path,array in sorted(tree.items()):
        a=np.asarray(array);h.update('/'.join(path).encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    assert read(HERE/'PREREGISTRATION.receipt.json')['preregistration_sha256']==sha(HERE/'PREREGISTRATION.md')
    assert read(HERE/'G0_REPORT.json')['status']=='PASS' and read(HERE/'G1A_REPORT.json')['status']=='PASS'
    root=ROOT/'diagnostics/four_way_intersection_stage1'
    ref=payload(root/'base_u_v13_broad_global_source_balanced_macflow/best.pkl')['config']
    checks=[];allhash={}
    for name in NAMES:
        for flavor in ('best','final'):
            if name.endswith('seed3111_superseded') and flavor=='final':continue
            ck=root/name/(flavor+'.pkl'); p=payload(ck);c=p['config']
            different=sorted(k for k in set(ref)|set(c) if str(ref.get(k))!=str(c.get(k)))
            permitted=NORMALIZERS|({'actor_hidden_dims'} if 'wide' in name else set())
            shape_ok=all(np.asarray(c[k]).shape==np.asarray(ref[k]).shape for k in NORMALIZERS)
            config_ok=set(different)<=permitted and c['environment_fingerprint']==ref['environment_fingerprint'] and shape_ok
            agent=FourFlowAdapter(ck)
            x=np.zeros((int(c['num_agents']),int(c['obs_dim'])),dtype=np.float32)
            key=jax.random.PRNGKey(271828)
            a=np.asarray(agent.native_fn(x,key));b=np.asarray(agent.native_fn(x,key))
            finite=bool(np.all(np.isfinite(a)))
            repeat=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
            fp32=(not jax.config.x64_enabled and a.dtype==np.float32)
            allhash[name+'|'+flavor]=fingerprint(p)
            item=dict(name=name,flavor=flavor,checkpoint_path=str(ck),checkpoint_sha256=sha(ck),
                      parameter_sha256=allhash[name+'|'+flavor],different_config_fields=different,
                      environment_fingerprint=c['environment_fingerprint'],normalizer_shapes_ok=shape_ok,
                      config_ok=config_ok,forward_finite=finite,forward_repeat_bytes_equal=repeat,
                      fp32=fp32,passed=config_ok and finite and repeat and fp32)
            checks.append(item)
            print(json.dumps({'gate':'G2','candidate':name+'|'+flavor,'passed':item['passed']}),flush=True)
    pair_ok=all(allhash[n+'|best']!=allhash[n+'|final'] for n in NAMES if not n.endswith('seed3111_superseded'))
    superseded_same=allhash['base_u_plus_w_v5_outer_buffer_dense_macflow_seed3111_superseded|best']==fingerprint(payload(root/'base_u_plus_w_v5_outer_buffer_dense_macflow_seed3111_superseded/final.pkl'))
    passed=len(checks)==25 and all(x['passed'] for x in checks) and pair_ok and superseded_same
    report=dict(gate='G2',status='PASS' if passed else 'FAIL',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                adapter_sha256=sha(HERE/'adapter.py'),reference_checkpoint_sha256=sha(root/'base_u_v13_broad_global_source_balanced_macflow/best.pkl'),
                expected=25,distinct_best_final_pairs=pair_ok,superseded_best_final_same=superseded_same,checks=checks)
    (HERE/'G2_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'gate':'G2','status':report['status'],'passed':sum(x['passed'] for x in checks),'required':25}),flush=True)
    if not passed:raise SystemExit('STOP: G2 failed')
if __name__=='__main__':main()
