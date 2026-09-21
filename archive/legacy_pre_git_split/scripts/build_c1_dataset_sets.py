"""Freeze all 200 Flow-BC train-pair starts and all 25 held-out val starts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.training.dataset_starts import load_dataset_starts
from single_integrator.environment import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026091302)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('frozen sets cannot be overwritten')
    root = ROOT / 'datasets/give_way_si_short_v1'
    plant = Config(**json.loads((root/'environment.json').read_text())['evaluation_environment'])
    train, train_meta = load_dataset_starts(root, 'train', plant)
    val, val_meta = load_dataset_starts(root, 'val', plant)
    if set(map(tuple, train.reshape(-1, 4))) & set(map(tuple, val.reshape(-1, 4))):
        raise ValueError('train and validation initial states overlap')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    streams = np.random.SeedSequence(args.seed).spawn(2)
    manifest = dict(protocol='flow_bc_dataset_initial_states_v1', seed=args.seed,
                    horizon=plant.max_steps, environment=plant.to_dict(), files={})
    for i, (name, starts, provenance) in enumerate((('calibration', train, train_meta), ('validation', val, val_meta))):
        path = args.out_dir/f'{name}.npz'
        noise = np.random.default_rng(streams[i]).standard_normal((len(starts), plant.max_steps, 4), dtype=np.float32)
        np.savez_compressed(path, initial_positions=starts, noise=noise,
                            metadata_json=json.dumps(provenance))
        manifest['files'][name] = dict(**provenance, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
