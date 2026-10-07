"""Source-selected input ablation on the same cached, already-opened cases."""
import argparse
import copy
import shutil
from pathlib import Path
from . import motion_counterexample_regression as regression
from .motion_factorial_train import OUT as MATRIX, SEEDS, read, write, sha

SOURCE = regression.OUT
OUT = MATRIX.parent/'motion_drop_rest_regression'


def prepare():
    if (OUT/'protocol.json').exists():
        return
    original = read(SOURCE/'protocol.json')
    selection_file = MATRIX.parent/'motion_drop_rest_control/source_selection.json'
    chosen = read(selection_file)['selection']['step']
    entries = []
    for fold in range(3):
        for seed in SEEDS:
            folder = MATRIX/'cv'/f'fold{fold}'/'motion_intervention'/'motion_only'/f'seed{seed}'
            read(folder/'complete.json')
            cp, norm = folder/f'step{chosen}.msgpack', folder/'normalization.json'
            entries.append(dict(arm='drop_rest', kind='motion_only', fold=fold, seed=seed, step=chosen,
                                path=str(folder), checkpoint=str(cp), checkpoint_sha256=sha(cp),
                                normalization=str(norm), normalization_sha256=sha(norm)))
    proto = copy.deepcopy(original)
    proto.update(experiment='posthoc_drop_rest_regression', models=entries,
                 source_selection_sha256=sha(selection_file), copied_source_protocol_sha256=sha(SOURCE/'protocol.json'))
    OUT.mkdir(exist_ok=True)
    paths = ['states.json', 'pairs.json', 'entities.npz']
    for profile in original['profiles']:
        paths += [f'inputs_{profile["name"]}.npz', f'audit_{profile["name"]}.json']
    copied = {}
    for name in paths:
        shutil.copyfile(SOURCE/name, OUT/name)
        assert sha(SOURCE/name) == sha(OUT/name)
        copied[name] = sha(OUT/name)
    proto['copied_input_hashes'] = copied
    write(OUT/'protocol.json', proto)


def evaluate():
    proto = read(OUT/'protocol.json')
    for name, digest in proto['copied_input_hashes'].items():
        assert sha(OUT/name) == digest
    regression.OUT = OUT
    regression.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'evaluate'))
    args = parser.parse_args()
    globals()[args.action]()
