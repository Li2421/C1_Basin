#!/usr/bin/env python3
"""Move prediction files to an archive with SHA256 checks and original-path links."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


def stamp(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def move_one(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = stamp(source)
    checksum = digest(source)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink() or digest(destination) != checksum:
            raise RuntimeError(f'Existing destination differs: {destination}')
    else:
        temporary = destination.with_name(destination.name + f'.copying-{os.getpid()}-{threading.get_ident()}')
        try:
            with temporary.open('xb') as output, source.open('rb') as incoming:
                shutil.copyfileobj(incoming, output, 1024 * 1024)
            shutil.copystat(source, temporary)
            if digest(temporary) != checksum:
                raise RuntimeError(f'Copied file checksum differs: {source}')
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    if stamp(source) != before:
        raise RuntimeError(f'Source changed during transfer: {source}')
    temporary_link = source.with_name(source.name + f'.archive-link-{os.getpid()}')
    try:
        temporary_link.symlink_to(destination)
        if temporary_link.resolve() != destination:
            raise RuntimeError(f'Incorrect link: {source}')
        os.replace(temporary_link, source)
    finally:
        temporary_link.unlink(missing_ok=True)
    return dict(source=str(source), destination=str(destination), bytes=before[2], sha256=checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='critic_fits directory')
    parser.add_argument('--archive', type=Path, required=True, help='dedicated archive directory')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    source, archive = args.source.resolve(), args.archive.resolve()
    if not source.is_dir() or archive.is_relative_to(source) or source.is_relative_to(archive):
        raise ValueError('Source and archive must be separate directories')
    archive.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (archive / '.migration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = sorted(source.glob('*/*/step*_predictions.npz'))
        if not files:
            raise RuntimeError('No prediction files found')
        tasks, existing = {}, 0
        for path in files:
            destination = archive / 'files' / path.relative_to(source)
            if path.is_symlink():
                if not destination.is_file() or path.resolve() != destination:
                    raise RuntimeError(f'Unexpected existing link: {path}')
                existing += 1
            else:
                tasks.setdefault(path.parent, []).append((path, destination))
        free = shutil.disk_usage(archive).free
        needed = sum(path.stat().st_size for group in tasks.values() for path, _ in group)
        if free < needed + 1024 ** 3:
            raise RuntimeError('Insufficient archive space')
        plan = dict(source=str(source), archive=str(archive), files=len(files),
                    already_linked=existing, bytes_to_move=needed,
                    utc=datetime.now(timezone.utc).isoformat())
        (archive / 'current_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        print(json.dumps(plan), flush=True)

        def group_move(group):
            return [move_one(path, target) for path, target in group]

        moved = size = 0
        with (archive / 'migration.jsonl').open('a') as log:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(group_move, group) for group in tasks.values()]
                try:
                    for index, future in enumerate(as_completed(futures), 1):
                        records = future.result()
                        for record in records:
                            log.write(json.dumps(record) + '\n')
                            moved += 1
                            size += record['bytes']
                        log.flush()
                        if index % 100 == 0:
                            print(json.dumps(dict(groups=index, total_groups=len(futures),
                                                  files=moved, GiB=round(size / 2 ** 30, 3),
                                                  seconds=round(time.monotonic() - started))), flush=True)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            os.fsync(log.fileno())
        subprocess.run(['sync', '-f', str(archive)], check=True)
        for path in files:
            destination = archive / 'files' / path.relative_to(source)
            if not path.is_symlink() or not destination.is_file() or path.resolve() != destination:
                raise RuntimeError(f'Final link verification failed: {path}')
        summary = dict(**plan, moved_files=moved, moved_bytes=size, verified_links=len(files),
                       seconds=round(time.monotonic() - started, 2), status='complete')
        (archive / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
