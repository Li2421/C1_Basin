"""Validate dynamics, provenance, pair separation and expert quality before training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.environment import Config


def validate_dataset(folder, require_full=True):
    folder=Path(folder)
    metadata=json.loads((folder/'environment.json').read_text())
    if metadata.get('schema')!='giveway_si_dataset_v1' or not metadata.get('complete'):
        raise ValueError('A complete, explicitly tagged SI dataset is required; legacy PID data is not accepted')
    config=Config(**metadata['environment'])
    source_fingerprint=hashlib.sha256(json.dumps(metadata['environment'],sort_keys=True).encode()).hexdigest()
    if source_fingerprint!=metadata['environment_fingerprint']:raise ValueError('Environment fingerprint mismatch')
    seen={};max_residual=0.;transitions=0;digest=hashlib.sha256()
    for file in sorted((folder/'raw').glob('*.npz')):
        digest.update(file.name.encode());digest.update(hashlib.sha256(file.read_bytes()).digest())
        with np.load(file,allow_pickle=False) as z:
            if str(z['environment_fingerprint'])!=source_fingerprint:raise ValueError(f'Mixed dynamics in {file}')
            obs,action,nxt=z['observations'],z['actions'],z['next_observations']
            if obs.shape!=(len(action),2,10) or action.shape!=(len(action),2,2) or nxt.shape!=obs.shape:raise ValueError('Unexpected shapes')
            if not all(np.isfinite(a).all() for a in [obs,action,nxt]):raise ValueError('Nonfinite transition')
            if np.linalg.norm(action,axis=-1).max()>config.max_speed+1e-7:raise ValueError('Unbounded expert action')
            residual=float(np.abs(nxt[:,:,:2].astype(float)-obs[:,:,:2].astype(float)-config.dt*action.astype(float)).max())
            max_residual=max(max_residual,residual)
            if residual>5e-7:raise ValueError(f'Transitions are not SI: {file}, residual={residual}')
            np.testing.assert_array_equal(nxt[:,:,2:4],action)
            np.testing.assert_array_equal(nxt[:-1],obs[1:])
            for data in [obs,nxt]:
                np.testing.assert_allclose(data[:,0,6:8],data[:,1,:2]-data[:,0,:2],atol=5e-7)
            if z['wall_collision'].any() or z['agent_collision'].any() or not bool(z['expert_complete']):raise ValueError('Expert quality gate failed')
            if not bool(z['dones'][-1]) or z['dones'][:-1].any():raise ValueError('Bad expert terminal labels')
            pair,mode=int(z['pair_id']),int(z['mode'])
            if (pair,mode) in seen:raise ValueError('Duplicate pair/mode')
            seen[pair,mode]=z['initial_positions'].copy()
            np.testing.assert_array_equal(obs[0,:,:2],z['initial_positions'])
            np.testing.assert_array_equal(obs[0,:,2:4],np.zeros((2,2)))
            transitions+=len(action)
    pairs={pair for pair,_ in seen}
    if not pairs:raise ValueError('Empty dataset')
    if require_full and pairs!=set(range(250)):raise ValueError('Baseline requires all 250 pairs')
    for pair in pairs:
        if (pair,0) not in seen or (pair,1) not in seen:raise ValueError('Unpaired modes')
        np.testing.assert_array_equal(seen[pair,0],seen[pair,1])
    result=dict(episodes=len(seen),pairs=len(pairs),transitions=transitions,max_float32_transition_residual=max_residual,dataset_sha256=digest.hexdigest(),environment_fingerprint=source_fingerprint)
    (folder/'validation.json').write_text(json.dumps(result,indent=2))
    return metadata,result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--allow_partial',action='store_true');a=p.parse_args()
    print(json.dumps(validate_dataset(a.dataset,not a.allow_partial)[1],indent=2))
