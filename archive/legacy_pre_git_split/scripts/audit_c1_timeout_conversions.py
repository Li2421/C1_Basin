"""Trace-only diagnosis; does not relabel outcomes or tune risk on these cases."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.environment import Config,GiveWayEnv


def metrics(trace,goals,dt,tolerance):
    positions=trace['positions_after'];velocity=trace['applied'].reshape(-1,2,2)
    errors=np.linalg.norm(positions-goals,axis=-1)
    result=dict(steps=len(positions),final_goal_errors=errors[-1].tolist(),
                agents_in_goal=int(np.sum(errors[-1]<=tolerance)))
    for seconds in (2,10):
        steps=min(len(positions),int(round(seconds/dt)))
        start=trace['positions_before'][-steps]
        progress=np.linalg.norm(start-goals,axis=-1)-errors[-1]
        speeds=np.linalg.norm(velocity[-steps:],axis=-1)
        path=speeds.sum(axis=0)*dt
        result[f'last_{seconds}s']=dict(actual_seconds=steps*dt,net_goal_progress=progress.tolist(),
            path_length=path.tolist(),max_speed=speeds.max(axis=0).tolist(),
            mean_speed=speeds.mean(axis=0).tolist(),
            progress_per_path=np.divide(progress,path,out=np.zeros_like(path),where=path>0).tolist())
    return result


def main():
    folder=ROOT/'results/c1_early_gradient_v2'
    protocol=json.loads((folder/'protocol.json').read_text())
    records=json.loads((folder/'records.json').read_text())
    index={(r['rid'],r['seed'],r['arm']):r for r in records}
    plant=Config(**protocol['environment']);goals=GiveWayEnv(plant).goals
    converted=[]
    for row in records:
        reference=index[row['rid'],row['seed'],'reference']
        if row['arm'] not in ('negative_early','negative_full') or not reference['either_deadlock'] or row['outcome']!='other_timeout':continue
        measured={};hashes={}
        for arm in ('reference',row['arm']):
            path=folder/f'{arm}_{row["rid"]}_{row["seed"]}.npz'
            hashes[arm]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as trace:measured[arm]=metrics(trace,goals,plant.dt,plant.goal_tolerance)
        converted.append(dict(rid=row['rid'],seed=row['seed'],arm=row['arm'],
                              metrics=measured,trace_hashes=hashes))
    result=dict(scope='posthoc mechanism audit; original labels unchanged',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                note='Path motion is not goal progress; different first-event lengths prevent causal comparison of final distances',
                converted=converted)
    (folder/'timeout_mechanism_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in converted:
        m=row['metrics'][row['arm']]
        print(json.dumps(dict(rid=row['rid'],seed=row['seed'],arm=row['arm'],
                              final_errors=m['final_goal_errors'],last_10s=m['last_10s'])))


if __name__=='__main__':main()
