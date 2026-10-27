#!/usr/bin/env python3
"""Compare upstream checkpoint actions to upstream saved states; no rollout."""
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
os.chdir(root)
sys.path.insert(0, str(root))
import numpy as np
import torch
import config
config.device = torch.device('cpu')
# Authors serialized CUDA tensors without map_location; preserve all weights on CPU.
_original_load = torch.load
def cpu_load(*args, **kwargs):
    kwargs['map_location'] = 'cpu'
    return _original_load(*args, **kwargs)
torch.load = cpu_load
from model_controller import ModelController

for name in ('Doorway', 'Intersection'):
    h = json.loads((root / 'experiment_results/histories' / f'LiveNet_{name}.json').read_text())
    steps = h['iterations']
    obs = np.array(h['obstacles'])
    controllers = [ModelController(f'weights/livenet_{name.lower()}_definition.json', np.array(steps[0]['goals'][i]), obs) for i in range(2)]
    errors = []
    for step in steps:
        states = np.array(step['states'])
        for i, controller in enumerate(controllers):
            controller.reset_state(states[i], states[1-i])
            prediction = controller.make_step(0, states[i]).ravel()
            prediction = np.clip(prediction, [-config.omega_limit, -config.accel_limit], [config.omega_limit, config.accel_limit])
            errors.append(np.abs(prediction - np.array(step['controls'][i])))
    errors = np.array(errors)
    print(json.dumps({'scenario': name, 'state_agent_pairs': len(errors), 'max_abs_error': errors.max(axis=0).tolist(), 'mean_abs_error': errors.mean(axis=0).tolist(), 'matched_at_1e-5': int(np.sum(np.max(errors, axis=1) < 1e-5))}))
