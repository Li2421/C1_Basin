"""Copy exactly compatible N=2/N=10 labels into the N=20 scaling design."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(prior: Path, current: Path) -> dict:
    old = json.loads((prior / 'design_manifest.json').read_text())
    new = json.loads((current / 'design_manifest.json').read_text())
    old_pool = json.loads((prior / 'eta_pool.json').read_text())
    new_pool = json.loads((current / 'eta_pool.json').read_text())
    if (old_pool['eta'] != new_pool['eta'] or
            old_pool['eta_uid'] != new_pool['eta_uid'] or
            old['future_indices'] != new['future_indices']):
        raise ValueError('global eta pool or future-seed protocol changed')
    counts = {}
    for n in (2, 10):
        a, b = old['scenario_info'][str(n)], new['scenario_info'][str(n)]
        if (a['scenario_uid'] != b['scenario_uid'] or
                a['controller_uid'] != b['controller_uid'] or
                a['controller'] != b['controller'] or
                a['source_manifest_sha256'] != b['source_manifest_sha256']):
            raise ValueError(f'N={n} scenario/controller identity changed')
        old_states = [s for s in old['states'] if s['N'] == n]
        new_states = [s for s in new['states'] if s['N'] == n]
        if old_states != new_states:
            raise ValueError(f'N={n} physical state identities changed')
        copied = observed = 0
        for state_index, state in enumerate(new_states):
            for eta_index, eta_uid in enumerate(new_pool['eta_uid']):
                index = state_index * len(new_pool['eta_uid']) + eta_index
                source = prior / f'rollouts/n{n}/pair_{index:04d}/result.json'
                if not source.exists():
                    continue
                result = json.loads(source.read_text())
                if (result['N'] != n or result['pair_index'] != index or
                        result['state_uid'] != state['state_uid'] or
                        result['eta_uid'] != eta_uid or
                        result['controller_uid'] != a['controller_uid'] or
                        result['flow_checkpoint_sha256'] !=
                        a['controller']['flow_checkpoint_sha256']):
                    raise ValueError(f'incompatible prior result: {source}')
                for future, row in result['seeds'].items():
                    if (int(future) not in new['future_indices'] or
                            row['state_uid'] != state['state_uid'] or
                            row['eta_uid'] != eta_uid or
                            row['controller_uid'] != a['controller_uid'] or
                            row['future_index'] != int(future)):
                        raise ValueError(f'incompatible prior rollout: {source} seed={future}')
                destination = current / f'rollouts/n{n}/pair_{index:04d}/result.json'
                if destination.exists():
                    raise FileExistsError(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if sha(source) != sha(destination):
                    raise RuntimeError(f'copy hash mismatch: {source}')
                copied += 1
                observed += len(result['seeds'])
        counts[str(n)] = {'copied_state_eta_pairs': copied,
                          'copied_seed_rollouts': observed}
    report = {'schema': 'gap1_exact_prior_result_reuse_v1',
              'prior_design': str(prior),
              'prior_design_sha256': sha(prior / 'design_manifest.json'),
              'current_design': str(current),
              'current_design_sha256': sha(current / 'design_manifest.json'),
              'identical_global_eta_pool': True,
              'identical_N2_N10_state_controller_and_seed_identities': True,
              'counts': counts}
    (current / 'prior_result_reuse.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=Path('datasets/gap1_eta_scaling_v2'))
    parser.add_argument('--current', type=Path, default=Path('datasets/gap1_eta_scaling_v3_n20'))
    args = parser.parse_args()
    print(json.dumps(run(args.prior, args.current), indent=2))


if __name__ == '__main__':
    main()
