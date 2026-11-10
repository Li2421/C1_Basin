#!/usr/bin/env python3
"""Relocate idle, fully checkpointed rollout SQLite and sidecars as one bundle."""
import json, os, shutil, sqlite3, subprocess
from pathlib import Path
from migrate_storage_20261110 import assert_idle, exchange, sha256, write_json

source=Path('/home/zhihan/research/Basin_C1/shared_rollout_db')
dest=Path('/mnt/ssd/zhihan/Basin_C1/shared_rollout_db/database')
audit=Path('/mnt/ssd/zhihan/Basin_C1/audits/storage_20261110/sqlite')
names=['rollout.sqlite','rollout.sqlite-wal','rollout.sqlite-shm']

def signature(p):
 s=p.stat(); return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def check(p,immutable=True):
 with sqlite3.connect(p.as_uri()+'?mode=ro'+('&immutable=1' if immutable else ''),uri=True) as con:
  result=con.execute('PRAGMA quick_check').fetchall()
  if result != [('ok',)]: raise RuntimeError(str(result))
  return con.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()

if not Path('/mnt/ssd').is_mount() or os.stat(source).st_dev==os.stat('/mnt/ssd').st_dev:
 raise RuntimeError('SSD not mounted')
assert_idle(source)
if any((source/n).is_symlink() or not (source/n).is_file() for n in names):
 raise RuntimeError('Expected three ordinary database files')
if (source/names[1]).stat().st_size != 0:
 raise RuntimeError('Nonempty WAL: retain source and use a separate database maintenance procedure')
if dest.exists(): raise RuntimeError('Destination already exists')
before={n:signature(source/n) for n in names}
schema=check(source/names[0])
audit.mkdir(parents=True,exist_ok=True); dest.mkdir(parents=True)
write_json(audit/'summary.json',{'status':'copying','before':before})
hashes={}
for n in names:
 shutil.copy2(source/n,dest/n)
 hashes[n]=sha256(source/n)
 if sha256(dest/n)!=hashes[n]: raise RuntimeError('Hash mismatch')
 print('verified',n,flush=True)
if check(dest/names[0]) != schema: raise RuntimeError('Schema mismatch')
assert_idle(source)
if {n:signature(source/n) for n in names} != before: raise RuntimeError('Source changed')
subprocess.run(['sync','-f',str(dest)],check=True)
write_json(audit/'sha256.json',hashes)
write_json(audit/'summary.json',{'status':'verified_before_switch','before':before})
exchanged=[]
try:
 for n in names:
  old=source/(n+'.migration-local-copy')
  if os.path.lexists(old): raise RuntimeError('Pending old copy exists')
  old.symlink_to(dest/n)
  exchange(source/n,old); exchanged.append(n)
 if check(source/names[0],immutable=False)!=schema: raise RuntimeError('Original-path SQLite read failed')
 # A read-only WAL connection can change SHM. The database itself must remain exact.
 if sha256(source/names[0])!=hashes[names[0]]: raise RuntimeError('Database changed')
except BaseException:
 for n in reversed(exchanged):
  old=source/(n+'.migration-local-copy'); exchange(source/n,old); old.unlink()
 raise
subprocess.run(['sync','-f',str(source)],check=True)
for n in names: (source/(n+'.migration-local-copy')).unlink()
write_json(audit/'summary.json',{'status':'complete','source':str(source),'destination':str(dest),'files':3,'bytes':sum(v[2] for v in before.values()),'sha256':hashes,'quick_check':'ok','original_path_read':'passed','wal_before_bytes':0})
print('complete',flush=True)
