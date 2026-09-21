"""Run the original Stage-I recipe for each preregistered larger geometry.

No C1 training or outcome-driven checkpoint selection occurs here. A completed
stage can be resumed only with matching source and input hashes. Partial stages
stop for inspection rather than silently overwrite evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    scenes = json.loads(args.scenes.read_text())['scenes']
    sources = ['scripts/run_c1_scene_stage_i.py', 'single_integrator/generate_scene.py',
        'single_integrator/diagnostics/uniform_state_data.py', 'single_integrator/train.py',
        'single_integrator/validate.py', 'single_integrator/environment.py',
        'single_integrator/expert.py', 'scripts/giveway_initial_state.py',
        'flowbc/train.py', 'flowbc/giveway_dataset.py', 'flowbc/giveway_flowbc_agent.py']
    protocol = dict(version='matched_scene_stage_i_v1',
        scenes_sha256=digest(args.scenes), scene_names=[s['name'] for s in scenes],
        source_sha256={s: digest(ROOT/s) for s in sources},
        pairs=250, data_seed=0, augmentation_seed=20260908,
        training_seeds=[0, 1], steps=100000, batch_size=256, normalize=True,
        selection='best_val.pkl: minimum fixed Stage-I validation flow loss; no C1 outcomes',
        training_initials='original sampler: x near +/-0.85, independent x noise +/-0.03 and y noise +/-0.002',
        evaluation='Matched initial distribution and wider initial distribution reported separately; paired Safety/C1 use identical frozen checkpoint, initials, noises and environment',
        architecture='Original Stage-I joint Flow-BC; no critic, guidance, recovery mixture, or stage weighting')
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Frozen protocol/source mismatch; choose a new version directory')
    else:
        if any(args.out.iterdir()):
            raise FileExistsError('Output must be empty when freezing protocol')
        write(path, protocol)
    if args.prepare_only:
        print(path, flush=True)
        return
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false')

    def run_stage(folder, name, command, outputs):
        marker = folder/(name+'.complete.json')
        log = folder/(name+'.log')
        if marker.exists():
            record = json.loads(marker.read_text())
            if record['command'] != command or record['outputs'] != {str(x): digest(x) for x in outputs}:
                raise ValueError(f'Completed stage changed: {marker}')
            return
        if log.exists():
            raise RuntimeError(f'Partial stage needs inspection: {log}')
        print(f'Start {folder.name}/{name}', flush=True)
        with log.open('x') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        write(marker, dict(command=command, outputs={str(x): digest(x) for x in outputs}))
        print(f'Complete {folder.name}/{name}', flush=True)

    for scene in scenes:
        folder = args.out/scene['name']
        folder.mkdir(exist_ok=True)
        plant = scene['environment']
        data, aug = folder/'dataset', folder/'augmented'
        run_stage(folder, 'generate', [sys.executable, '-u', '-m', 'single_integrator.generate_scene',
            '--out_dir', str(data), '--seed', '0', '--pairs', '250',
            '--corridor_half_length', str(plant['corridor_half_length']),
            '--corridor_width', str(plant['corridor_width']), '--bay_top', str(plant['bay_top'])],
            [data/'environment.json'])
        # The generator intentionally varies only these three geometric inputs.
        metadata = json.loads((data/'environment.json').read_text())
        if metadata['evaluation_environment'] != plant:
            raise ValueError('Generated task differs from preregistered environment')
        run_stage(folder, 'augment', [sys.executable, '-u', '-m',
            'single_integrator.diagnostics.uniform_state_data', '--dataset', str(data),
            '--out_dir', str(aug), '--seed', '20260908'], [aug/'environment.json'])
        for seed in (0, 1):
            out = folder/f'seed{seed}'
            run_stage(folder, f'train_seed{seed}', [sys.executable, '-u', '-m',
                'single_integrator.train', '--dataset', str(data), '--training_data_dir', str(aug/'raw'),
                '--out_dir', str(out), '--steps', '100000', '--batch_size', '256', '--seed', str(seed)],
                [out/'best_val.pkl', out/'benchmark.json', out/'metrics.jsonl'])
            frozen = folder/f'frozen_seed{seed}.json'
            record = dict(checkpoint=str((out/'best_val.pkl').resolve()),
                sha256=digest(out/'best_val.pkl'), scene=scene['name'], environment=plant,
                selection=protocol['selection'], stage_i_protocol_sha256=digest(path))
            if frozen.exists() and json.loads(frozen.read_text()) != record:
                raise ValueError('Frozen baseline changed')
            write(frozen, record)
    write(args.out/'complete.json', dict(scenes=len(scenes), seeds=[0, 1]))


if __name__ == '__main__':
    main()
