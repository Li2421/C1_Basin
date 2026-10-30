#!/usr/bin/env python3
"""Copy static JSON/NPZ/PKL result directories, verify SHA256, and retain old paths.

Linux renameat2 exchanges each source with its destination symlink atomically.
Only the verified local copy is removed. Code, databases, symlinks, hard links,
recent files, and directories held open by a process are rejected.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time


def write_json(path, value):
    with path.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(root, min_age_hours):
    result = {}
    cutoff = time.time() - min_age_hours * 3600
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in ['.'] + dirs + files:
            path = Path(directory) if name == '.' else Path(directory) / name
            info = path.lstat()
            relative = str(path.relative_to(root))
            if stat.S_ISDIR(info.st_mode):
                kind = 'directory'
            elif stat.S_ISREG(info.st_mode):
                kind = 'file'
                if path.suffix not in {'.json', '.npz', '.pkl'} or info.st_nlink != 1:
                    raise RuntimeError(f'Not a standalone JSON/NPZ/PKL result: {path}')
                if info.st_mtime > cutoff:
                    raise RuntimeError(f'Result is too recent: {path}')
            else:
                raise RuntimeError(f'Unsupported file or link: {path}')
            result[relative] = [kind, info.st_dev, info.st_ino, info.st_size,
                                info.st_mtime_ns, info.st_ctime_ns, info.st_mode]
    return result


def assert_idle(root):
    prefix = str(root) + '/'
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            handles = [process / 'cwd'] + list((process / 'fd').iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for handle in handles:
            try:
                target = os.readlink(handle)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if target == str(root) or target.startswith(prefix):
                raise RuntimeError(f'Process {process.name} holds {target}')


def exchange(left, right):
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                       ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(left), -100, os.fsencode(right), 2):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def run_one(source, destination, audit, min_age_hours, workers):
    audit.mkdir(parents=True, exist_ok=True)
    summary_path = audit / 'summary.json'
    if source.is_symlink():
        if source.resolve() != destination or not summary_path.exists():
            raise RuntimeError(f'Unexpected migrated source: {source}')
        previous = json.loads(summary_path.read_text())
        if previous.get('status') != 'complete':
            raise RuntimeError(f'Incomplete previous switch; inspect {audit}')
        return previous
    if not source.is_dir() or source.resolve() != source:
        raise RuntimeError(f'Source must be an ordinary directory: {source}')
    if destination.is_symlink():
        raise RuntimeError(f'Destination must not be a symlink: {destination}')
    assert_idle(source)
    before = inventory(source, min_age_hours)
    files = sorted(name for name, row in before.items() if row[0] == 'file')
    total_bytes = sum(before[name][3] for name in files)
    if not files:
        raise RuntimeError(f'Empty source: {source}')
    if shutil.disk_usage(audit).free < total_bytes + 1024 ** 3:
        raise RuntimeError('Insufficient destination space')
    summary = dict(source=str(source), destination=str(destination), files=len(files),
                   bytes=total_bytes, status='copying',
                   utc=datetime.now(timezone.utc).isoformat())
    write_json(summary_path, summary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(summary), flush=True)
    subprocess.run(['rsync', '-a', '--', str(source) + '/', str(destination) + '/'],
                   check=True)
    # Compare every entry; destination inode/ctime naturally differ after copying.
    copied = inventory(destination, 0)
    if set(copied) != set(before):
        raise RuntimeError('Destination entry set differs')
    for name, row in before.items():
        other = copied[name]
        if row[0] != other[0] or row[6] != other[6]:
            raise RuntimeError(f'Destination kind/mode differs: {name}')
        if row[0] == 'file' and (row[3] != other[3] or row[4] != other[4]):
            raise RuntimeError(f'Destination size/mtime differs: {name}')

    def verify(name):
        original = sha256(source / name)
        if sha256(destination / name) != original:
            raise RuntimeError(f'SHA256 mismatch: {name}')
        return dict(path=name, bytes=before[name][3], sha256=original)

    progress = time.monotonic()
    with (audit / 'sha256.jsonl').open('w') as manifest:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, record in enumerate(pool.map(verify, files), 1):
                manifest.write(json.dumps(record) + '\n')
                if time.monotonic() - progress >= 30:
                    print(json.dumps(dict(source=str(source), verified=index,
                                          total=len(files))), flush=True)
                    progress = time.monotonic()
        manifest.flush()
        os.fsync(manifest.fileno())
    assert_idle(source)
    if inventory(source, min_age_hours) != before:
        raise RuntimeError('Source changed during copy/verification; source retained')
    if inventory(destination, 0) != copied:
        raise RuntimeError('Destination changed during verification; source retained')
    subprocess.run(['sync', '-f', str(destination)], check=True)
    summary['status'] = 'verified_before_switch'
    write_json(summary_path, summary)
    backup = source.with_name(source.name + '.migration-local-copy')
    if os.path.lexists(backup):
        raise RuntimeError(f'Previous switch artifact exists: {backup}')
    backup.symlink_to(destination, target_is_directory=True)
    try:
        exchange(source, backup)
    except BaseException:
        backup.unlink()
        raise
    try:
        assert_idle(backup)
        after_switch = inventory(backup, min_age_hours)
        # Renaming the root directory changes its ctime, but not its contents.
        after_switch['.'][5] = before['.'][5]
        if after_switch != before or source.resolve() != destination:
            raise RuntimeError('Post-switch source/link check failed')
    except BaseException:
        exchange(source, backup)
        backup.unlink()
        raise
    summary['status'] = 'switched_local_copy_pending_removal'
    write_json(summary_path, summary)
    subprocess.run(['sync', '-f', str(source.parent)], check=True)
    shutil.rmtree(backup)
    summary['status'] = 'complete'
    summary['verified_sha256_files'] = len(files)
    write_json(summary_path, summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan['source_root']).resolve()
    destination_root = Path(plan['destination_root']).resolve()
    audit_root = Path(plan['audit_root']).resolve()
    if root.is_relative_to(destination_root) or destination_root.is_relative_to(root):
        raise RuntimeError('Source and destination roots must be separate')
    if os.stat(root).st_dev == os.stat(audit_root).st_dev:
        raise RuntimeError('Destination must be on another filesystem')
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'])
    tracked = tracked.decode().split('\0')
    sources = []
    for relative in plan['directories']:
        source = root / relative
        if '..' in Path(relative).parts or Path(relative).is_absolute():
            raise RuntimeError('Invalid relative path')
        if any(name == relative or name.startswith(relative + '/') for name in tracked):
            raise RuntimeError(f'Directory contains Git-tracked files: {relative}')
        if any(source.is_relative_to(old) or old.is_relative_to(source) for old in sources):
            raise RuntimeError('Overlapping source directories')
        sources.append(source)
    if not args.apply:
        print(json.dumps(plan, indent=2))
        return
    with (audit_root / '.migration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for source in sources:
            relative = source.relative_to(root)
            rows.append(run_one(source, destination_root / relative, audit_root / relative,
                                plan.get('min_age_hours', 24), args.workers))
        write_json(audit_root / 'summary.json', dict(status='complete', directories=rows,
                   files=sum(row['files'] for row in rows),
                   bytes=sum(row['bytes'] for row in rows)))


if __name__ == '__main__':
    main()
