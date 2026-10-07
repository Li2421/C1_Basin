"""Reuse the frozen scalar evaluator, altering task inventory and metadata only."""
import hashlib
import json
import sys
import types
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
source_path = ROOT / 'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/run_rollouts.py'
protocol = json.loads((OUT / 'protocol.json').read_text())
source = source_path.read_text()
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == protocol['runner_sha256']
basis_path = ROOT / 'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'
assert hashlib.sha256(basis_path.read_bytes()).hexdigest() == protocol['basis_sha256']
old = '"state_id": f"wide_ic_frozen_{ep:04d}", "episode_index": ep,'
new = '"state_id": state["state_id"], "episode_index": ep, "parent_episode_index": state["parent_episode_index"], "h0": state["h_raw"], "physical_offset_m": state["physical_offset_m"],'
assert source.count(old) == 1
source = source.replace(old, new)
runner = types.ModuleType('frozen_scalar_local_probe_runner')
runner.__file__ = str(source_path)
sys.modules[runner.__name__] = runner
exec(compile(source, str(source_path), 'exec'), runner.__dict__)
runner.OUT = OUT
runner.EXPERIMENT_UID = protocol['experiment_uid']

def tasks():
    states = json.loads((OUT / 'frozen_proposals.json').read_text())['states']
    return [{'state': s, 'kind': kind, 'aliases': [kind], 'eta': eta, 'future_index': seed}
            for s in states for kind, eta in s['eta'].items() for seed in range(16)]

runner.tasks = tasks
if __name__ == '__main__': runner.main()
