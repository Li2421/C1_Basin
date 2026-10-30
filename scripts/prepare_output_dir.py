#!/usr/bin/env python3
"""Prepare a new pure-artifact directory on SSD, retaining its repository path."""
import argparse
import os
from pathlib import Path
import subprocess


def prepare(relative):
    repo = Path(__file__).resolve().parents[1]
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or len(rel.parts) < 2:
        raise ValueError('Use a repository-relative artifact subdirectory')
    if rel.parts[0] not in {'diagnostics', 'outputs', 'datasets', 'results'}:
        raise ValueError('Only artifact locations are supported')
    disk = Path('/mnt/ssd')
    if not disk.is_mount() or disk.stat().st_dev == repo.stat().st_dev:
        raise RuntimeError('Data SSD is not mounted; refusing home fallback')
    target = disk / 'zhihan/Basin_C1' / rel
    source = repo / rel
    if target.resolve() != target or source.parent.resolve() != source.parent:
        raise RuntimeError('Unexpected symlink in parent paths')
    tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z'], text=True).split('\0')
    if any(n == str(rel) or n.startswith(str(rel) + '/') for n in tracked):
        raise RuntimeError('Location contains tracked files')
    if source.is_symlink():
        if source.resolve() != target or not target.is_dir():
            raise RuntimeError('Unexpected or broken existing link')
        return target
    if os.path.lexists(source):
        raise RuntimeError('Existing source requires verified migration')
    target.mkdir(parents=True, exist_ok=True)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.symlink_to(target, target_is_directory=True)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('relative', nargs='+')
    for relative in parser.parse_args().relative:
        print(prepare(relative))
