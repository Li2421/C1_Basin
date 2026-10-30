"""Official database identity plus experiment-private numerical imports."""
import guard
from guard import HERE, ROOT
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

assert (HERE / 'PRIOR_TASK_DELIVERED.json').exists(), 'Numerical imports stay queued until prior study delivery.'

# The source and PoC trees contain shadow database packages. Pin the official
# package before introducing any archived execution namespace.
sys.path.insert(0, str(ROOT))
import shared_rollout_db
from shared_rollout_db.src import rollout_db as db, planner, cache_writer
assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT / 'shared_rollout_db')
assert cache_writer.ROOT.resolve() == ROOT / 'shared_rollout_db'

PRIVATE = HERE / 'private/Basin_C1'
sys.path.insert(0, str(PRIVATE))
for package in ('shared_control', 'single_integrator', 'bottleneck_family', 'new_benchmark_common', 'diagnostics'):
    module = importlib.import_module(package)
    assert Path(module.__file__).resolve().is_relative_to(PRIVATE), (package, module.__file__)
importlib.import_module('new_benchmark_common.macflow')
importlib.import_module('diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry')
sys.path.insert(0, str(HERE / 'generated'))
import bootstrap
import runtime_new as RN
assert Path(bootstrap.__file__).resolve().is_relative_to(HERE/'generated')
assert Path(RN.__file__).resolve().is_relative_to(HERE/'generated')
import numpy as np

EXP = 'exp_paper_baseline_protocol_v1_20261029'


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(db.canonical(clean(value)).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def controllers():
    selected = read(HERE / 'inputs/diagnostics/cross_scene_correction_v1/SELECTED_CONTROLLERS.json')['xsc_swap_n6']
    return [selected[key] for key in ('centralized_sampling', 'gradient_mpc')]
