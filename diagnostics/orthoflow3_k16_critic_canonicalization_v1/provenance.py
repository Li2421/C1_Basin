"""Local artifact provenance only; not a general DB-integrity audit."""
from datetime import datetime,timezone
from pathlib import Path
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a
from new_benchmark_common.safety_eta3 import SCENARIOS,sha256_sources
from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as db


def snapshot():
    paths=[db.CHECKPOINT,a.ROOT/'shared_control/basis_families.py',a.ROOT/'shared_control/hard_projection.py',
           a.ROOT/'diagnostics/double_bottleneck_eta3_basin/tools/exact_projection_retry.py']
    for spec in SCENARIOS.values():paths += [spec['checkpoint'],spec['dataset']/'manifest.json']
    result={'recorded_at':datetime.now(timezone.utc).isoformat(),'files':{str(p.relative_to(a.ROOT)):a.sha(p) for p in paths},
            'eta_domain':{'low':a.learn.LOW.tolist(),'high':a.learn.HIGH.tolist(),'network_normalization':'subtract center,divide half-width'},
            'basis_source_hash':a.sha(a.ROOT/'shared_control/basis_families.py'),
            'double_expected_hashes_verified':all(a.sha(p)==h for p,h in db.EXPECTED.items()),
            'Ring_safety_hash':sha256_sources([a.ROOT/'shared_control/hard_projection.py',a.ROOT/'ring_exchange/safety.py',
                                              a.ROOT/'diagnostics/double_bottleneck_eta3_basin/tools/exact_projection_retry.py']),
            'rollout_jobs':{'245states':1442,'initial_expansion':1461,'collect':1462,'critic_training':1463},
            'no_canonical_modifications':True,'phase_B_started':False,
            'historical_runtime_field_note':'TrainingRuntime raw provenance_experiment may name basin_dataset_v1; task-specific DB experiment_uid,source_file,stage and proposal_set_hash identify these new acquisitions.'}
    a.dump('runtime_artifacts.json',result)
    return result


if __name__=='__main__':
    print(a.canonical(snapshot()))
