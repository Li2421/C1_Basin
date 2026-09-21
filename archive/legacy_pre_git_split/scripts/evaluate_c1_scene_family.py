"""Predeclared geometry-input transfer, with unchanged learned control path."""
import argparse
import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.train_deadlock_primary import setup, noise, digest
from single_integrator.c1.evaluate_deadlock_primary import execute
from single_integrator.c1.risk.deadlock_primary import trajectory
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.environment import Config, GiveWayEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--baseline-seed', type=int, choices=(0, 1), required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--split', choices=('development', 'test'), default='development')
    p.add_argument('--limit-per-scene', type=int)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.split=='test' and (args.limit_per_scene is not None or args.checkpoint is None):
        raise ValueError('final evaluation requires selected checkpoint and complete scene pool')
    if args.limit_per_scene is not None and args.limit_per_scene < 1:
        raise ValueError('limit must be positive')
    data = json.loads(args.scenes.read_text())
    if data['version']!='c1_scene_family_v1':
        raise ValueError('unknown scene protocol')
    baseline, field, _, cbf, basepath = setup(baseline_seed=args.baseline_seed)
    methods = [('Safety', baseline)]
    if args.checkpoint:
        saved = pickle.loads(args.checkpoint.read_bytes())
        config = saved['config']
        if config['version']!='c1_deadlock_primary_v1' or config['baseline_sha256']!=digest(basepath):
            raise ValueError('checkpoint and frozen baseline mismatch')
        if config['objective']!='constrained' or 'selection' not in saved or saved['selection']['constraint'] > 0:
            raise ValueError('transfer requires feasible selected constrained checkpoint')
        if any(digest(ROOT/path)!=sha for path,sha in config['source_hashes'].items()):
            raise ValueError('checkpoint source provenance mismatch')
        methods.append(('C1', jax.tree_util.tree_map(jnp.asarray, saved['params'])))
    args.out.mkdir(parents=True)
    atomic_save(args.out/'protocol.json', dict(scene_sha256=digest(args.scenes),
        baseline_seed=args.baseline_seed, baseline_sha256=digest(basepath),
        checkpoint_sha256=digest(args.checkpoint) if args.checkpoint else None,
        split=args.split, limit_per_scene=args.limit_per_scene,
        zero_shot=True, backend=jax.default_backend(), evaluator_sha256=digest(Path(__file__))))
    summary = {}
    for scene in data['scenes']:
        plant = Config(**scene['environment'])
        goals = jnp.asarray(GiveWayEnv(plant).goals)
        score = jax.jit(lambda x,y,u,a: trajectory(x,y,u,goals,a,dt=plant.dt,
            max_speed=plant.max_speed,goal_tolerance=plant.goal_tolerance,
            hold_seconds=plant.deadlock_hold_seconds,progress_window_seconds=plant.progress_window_seconds,
            progress_epsilon=plant.progress_epsilon,speed_epsilon_fraction=plant.speed_epsilon_fraction)['J_live'])
        pool = scene['pools'][args.split]
        if args.limit_per_scene:
            pool = pool[:args.limit_per_scene]
        folder = args.out/scene['name']
        folder.mkdir()
        rows = []
        for item in pool:
            for seed in data[args.split+'_noise_seeds']:
                draws = noise(seed, item['rid'])
                for name, params in methods:
                    row, trace = execute(params, field, item['initial'], draws, plant, cbf)
                    n = row['steps']; pad = plant.max_steps-n
                    x = np.concatenate([trace['positions_before'], np.repeat(trace['positions_after'][-1:],pad,axis=0)])
                    y = np.concatenate([trace['positions_after'], np.repeat(trace['positions_after'][-1:],pad,axis=0)])
                    u = np.concatenate([trace['applied'], np.zeros((pad,4))])
                    row.update(method=name,rid=item['rid'],noise_seed=seed,scene=scene['name'],
                        J_live=float(score(jnp.asarray(x),jnp.asarray(y),jnp.asarray(u),jnp.arange(plant.max_steps)<n)))
                    stem = f'{name}_{item["rid"]}_{seed}'
                    atomic_save(folder/(stem+'.json'),row)
                    np.savez_compressed(folder/(stem+'.npz'),**trace)
                    rows.append(row)
        summary[scene['name']] = {name:dict(n=sum(r['method']==name for r in rows),
            **{key:sum(r[key] for r in rows if r['method']==name) for key in ('success','any_deadlock','timeout')},
            J_live=float(np.mean([r['J_live'] for r in rows if r['method']==name]))) for name,_ in methods}
        atomic_save(folder/'complete.json',dict(episodes=len(rows)))
        atomic_save(args.out/'summary.json',summary)
        print(dict(scene=scene['name'],result=summary[scene['name']]),flush=True)
    atomic_save(args.out/'complete.json',dict(scenes=len(summary),scope=args.split))


if __name__ == '__main__':
    main()
