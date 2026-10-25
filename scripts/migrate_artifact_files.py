#!/usr/bin/env python3
"""Migrate selected static files in mixed artifact directories with SHA256 checks."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

from migrate_result_directories import assert_idle, sha256, write_json


def stamp(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f'Expected standalone regular file: {path}')
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode]


def run_job(job, workers):
    source = Path(job['source'])
    target = Path(job['target'])
    audit = Path(job['audit'])
    pattern = job['pattern']
    if pattern not in {'*/*/step*.msgpack', '*/*/step*_predictions.npz'}:
        raise RuntimeError('Only explicitly scoped critic checkpoint/prediction patterns allowed')
    if source.resolve() != source or not source.is_dir():
        raise RuntimeError('Source must be an ordinary absolute directory')
    if target.resolve() != target or audit.resolve() != audit:
        raise RuntimeError('Destination/audit path must not contain symlinks')
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise RuntimeError('Source and destination must be separate')
    audit.mkdir(parents=True, exist_ok=True)
    if os.stat(source).st_dev == os.stat(audit).st_dev:
        raise RuntimeError('Destination must be on a different filesystem')
    with (audit / '.migration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(job, source, target, audit, pattern, workers)


def run_locked(job, source, target, audit, pattern, workers):
    summary_path = audit / 'summary.json'
    files = sorted(source.glob(pattern))
    if not files:
        raise RuntimeError('No matching files')
    if len(files) != job.get('expected_files', len(files)):
        raise RuntimeError('Selected file count differs from audited plan')
    if any(path.is_symlink() for path in files):
        if not summary_path.exists():
            raise RuntimeError('Unexpected selected symlink; source retained')
        previous = json.loads(summary_path.read_text())
        if previous.get('status') != 'complete' or previous['files'] != len(files):
            raise RuntimeError('Previous switch incomplete; recover using durable SHA256 manifest')
        for path in files:
            expected = target / path.relative_to(source)
            if not path.is_symlink() or path.resolve() != expected or not expected.is_file():
                raise RuntimeError(f'Existing link verification failed: {path}')
        return previous
    assert_idle(source)
    before = {str(path.relative_to(source)): stamp(path) for path in files}
    cutoff = time.time() - job.get('min_age_hours', 24) * 3600
    if any(row[3] / 1e9 > cutoff for row in before.values()):
        raise RuntimeError('Selected file is too recent')
    repo = Path(job['repository'])
    tracked = set(subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z'],
                                         text=True).split('\0'))
    if any(str(path.relative_to(repo)) in tracked for path in files):
        raise RuntimeError('Selection contains tracked files')
    total = sum(row[2] for row in before.values())
    if total != job.get('expected_bytes', total):
        raise RuntimeError('Selected byte count differs from audited plan')
    if shutil.disk_usage(audit).free < total + 1024 ** 3:
        raise RuntimeError('Insufficient destination space')
    summary = dict(name=job['name'], source=str(source), target=str(target),
                   pattern=pattern, files=len(files), bytes=total, status='copying',
                   utc=datetime.now(timezone.utc).isoformat())
    write_json(summary_path, summary)
    print(json.dumps(summary), flush=True)
    # rsync receives only the selected regular files; all other source entries stay put.
    listing = audit / 'files.from0'
    with listing.open('wb') as stream:
        for name in before:
            stream.write(os.fsencode(name) + b'\0')
        stream.flush()
        os.fsync(stream.fileno())
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '--from0', '--files-from=' + str(listing), '--',
                    str(source) + '/', str(target) + '/'], check=True)
    copied = {name: stamp(target / name) for name in before}
    for name, row in before.items():
        other = copied[name]
        if [row[2], row[3], row[5]] != [other[2], other[3], other[5]]:
            raise RuntimeError(f'Destination size, mtime or mode differs: {name}')
    if {str(p.relative_to(target)) for p in target.glob(pattern)} != set(before):
        raise RuntimeError('Destination selected file set differs')

    def verify(name):
        checksum = sha256(source / name)
        if sha256(target / name) != checksum:
            raise RuntimeError(f'SHA256 mismatch: {name}')
        return dict(path=name, bytes=before[name][2], sha256=checksum)

    progress = time.monotonic()
    with (audit / 'sha256.jsonl').open('w') as manifest:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, record in enumerate(pool.map(verify, before), 1):
                manifest.write(json.dumps(record) + '\n')
                if time.monotonic() - progress >= 30:
                    print(json.dumps(dict(phase='verify', files=index, total=len(files))), flush=True)
                    progress = time.monotonic()
        manifest.flush()
        os.fsync(manifest.fileno())
    assert_idle(source)
    if set(source.glob(pattern)) != set(files):
        raise RuntimeError('Source selection changed; source retained')
    for name, row in before.items():
        if stamp(source / name) != row or stamp(target / name) != copied[name]:
            raise RuntimeError(f'File changed during verification: {name}')
    subprocess.run(['sync', '-f', str(target)], check=True)
    summary['status'] = 'verified_before_switch'
    write_json(summary_path, summary)
    # Both the complete destination and recovery manifest are durable before unlinking.
    subprocess.run(['sync', '-f', str(audit)], check=True)
    progress = time.monotonic()
    for index, (name, row) in enumerate(before.items(), 1):
        path = source / name
        destination = target / name
        if stamp(path) != row or stamp(destination) != copied[name]:
            raise RuntimeError(f'File changed immediately before switching: {path}')
        temporary = path.with_name(path.name + f'.migration-link-{os.getpid()}')
        try:
            temporary.symlink_to(destination)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        if time.monotonic() - progress >= 30:
            print(json.dumps(dict(phase='link', files=index, total=len(files))), flush=True)
            progress = time.monotonic()
    subprocess.run(['sync', '-f', str(source)], check=True)
    for path in files:
        destination = target / path.relative_to(source)
        if not path.is_symlink() or path.resolve() != destination or not destination.is_file():
            raise RuntimeError(f'Final link verification failed: {path}')
    summary.update(status='complete', verified_sha256_files=len(files), verified_links=len(files))
    write_json(summary_path, summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    jobs = json.loads(args.plan.read_text())['jobs']
    job = next(row for row in jobs if row['name'] == args.job)
    if args.apply:
        run_job(job, args.workers)
    else:
        print(json.dumps(job, indent=2))


if __name__ == '__main__':
    main()
