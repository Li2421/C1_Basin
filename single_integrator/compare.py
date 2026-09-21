"""Check the causal comparison contract before reporting paired outcomes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def compare(left,right,expect_identity=False):
    left,right=Path(left),Path(right)
    configs=[json.loads((p/'config.json').read_text()) for p in [left,right]]
    for key in ['checkpoint_sha256','environment_fingerprint','seed','split','n_rollouts','rng_protocol','plant_source_sha256','runtime_code_sha256']:
        if configs[0][key]!=configs[1][key]:raise ValueError(f'Confounded comparison: {key} differs')
    results=[json.loads((p/'summary.json').read_text()) for p in [left,right]]
    if len(results[0]['rollouts'])!=len(results[1]['rollouts']):raise ValueError('Unpaired rollouts')
    rows=[];max_residual=0.
    for a,b in zip(results[0]['rollouts'],results[1]['rollouts']):
        if a['rollout_id']!=b['rollout_id'] or a['pair_id']!=b['pair_id']:raise ValueError('Pair IDs differ')
        np.testing.assert_array_equal(a['initial_positions'],b['initial_positions'])
        rid=a['rollout_id']
        with np.load(left/f'rollout_{rid:04d}.npz') as aa,np.load(right/f'rollout_{rid:04d}.npz') as bb:
            for z,c in zip([aa,bb],configs):
                dt=c['environment']['dt']
                residual=np.abs(z['positions']-z['positions_before']-dt*z['executed_velocity']).max()
                max_residual=max(max_residual,float(residual))
                if residual>1e-12:raise ValueError('Executed trajectory is not the declared integrator')
                np.testing.assert_array_equal(z['velocities'],z['executed_velocity'])
                if np.linalg.norm(z['executed_velocity'],axis=-1).max()>c['environment']['max_speed']+1e-9:raise ValueError('Overspeed executed action')
            if expect_identity:
                for key in ['observations','raw_policy_velocity','nominal_velocity','executed_velocity','positions','velocities','wall_collision','agent_collision','deadlock','window_progress','stuck_timer','candidate_deadlock','deadlock_trigger_timestep']:
                    np.testing.assert_array_equal(aa[key],bb[key])
        row=dict(rollout_id=rid,pair_id=a['pair_id'])
        for metric in ['success','collision_free_success','wall_collision','agent_collision','deadlock']:
            row['left_'+metric]=a[metric];row['right_'+metric]=b[metric]
        rows.append(row)
    return dict(causal_contract_passed=True,identity_verified=expect_identity,n_pairs=len(rows),max_integration_residual=max_residual,left_aggregate=results[0]['aggregate'],right_aggregate=results[1]['aggregate']),rows


def main():
    p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('--expect_identity',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report,rows=compare(a.left,a.right,a.expect_identity)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2))
    with a.out.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
