"""Freeze task geometry variations; supplies no policy or navigation actions."""
import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.cbf import CBFConfig
from single_integrator.c1.training.persistence import atomic_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    base = Config(**json.loads((ROOT/'results/c1_deadlock_primary/sets.json').read_text())['environment'])
    rng = np.random.default_rng(2026091673)
    scenes = []
    clearance = base.agent_radius+base.wall_radius+base.wall_collision_margin+CBFConfig().separation_buffer
    separation = 2*base.agent_radius+base.agent_collision_margin+CBFConfig().separation_buffer
    for index, (length, width, top) in enumerate(itertools.product((1.8, 2.5), (.4, .48), (.68, .8))):
        plant = replace(base, corridor_half_length=length, corridor_width=width, bay_top=top)
        env = GiveWayEnv(plant)
        # Geometric feasibility checks, not a generated route or expert policy:
        # individual center clearance exists in the corridor; side space can
        # hold one center farther from centerline than pair safety separation.
        if width/2 <= clearance or top-clearance <= separation:
            raise ValueError('no certified center clearance / side holding space')
        pools = {}
        for split, n in (('development', 32), ('test', 256)):
            x = rng.uniform(.55/1.3, 1.05/1.3, (n, 2))*length
            y = rng.uniform(-.025, .025, (n, 2))
            starts = np.stack([np.c_[-x[:, 0], y[:, 0]], np.c_[x[:, 1], y[:, 1]]], axis=1)
            for initial in starts:
                env.reset(initial)
            offset = 0 if split=='development' else 32
            pools[split] = [dict(rid=20000+index*1000+offset+i, initial=initial.tolist())
                           for i, initial in enumerate(starts)]
        scenes.append(dict(name=f'L{length}_W{width}_B{top}', environment=plant.to_dict(), pools=pools,
            geometry_checks=dict(center_half_width=width/2-clearance,
                holding_clearance=top-clearance-separation)))
    protocol = dict(version='c1_scene_family_v1', seed=2026091673,
        purpose='Frozen geometry-input transfer study, no scene-specific control or loss changes',
        baseline='Same two original frozen checkpoints for Safety and C1; out-of-training-domain transfer',
        development_noise_seeds=[72401, 72402], test_noise_seeds=[72501, 72502, 72503, 72504],
        primary='first-event strict deadlock resolved into success; timeout and total failures separate',
        selection='All eight geometries predeclared; no selection by favorable baseline or C1 outcomes',
        transfer='Zero-shot evaluation of the small-scene selected residual; no fine tuning',
        comparison_family_size=48, scenes=scenes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(args.out, protocol)
    print(dict(path=str(args.out), scenes=len(scenes), sha256=hashlib.sha256(args.out.read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
