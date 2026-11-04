"""Verified relocation of oversized, unexecuted preparation metadata only.

Run only after this code/protocol has been committed and pushed. Active task
files, controller sources, SQLite/WAL/SHM and all rollout evidence are untouched.
"""
from native_io import *
import shutil
def main():
    assert Path('/mnt/ssd').is_mount()
    active={c['rollout_uid'] for c in read(D/'ACTIVE_NATIVE_CASES.json')}
    files=[D/'ACTIVE_CASES.json',D/'tasks/native_planned.json']
    files += [p for p in (D/'tasks').glob('roll_*.json') if p.stem not in active and not p.is_symlink()]
    dest=D/'artifacts/superseded_preparation_metadata';dest.mkdir(parents=True,exist_ok=True);records=[]
    for p in files:
        if p.is_symlink() or not p.exists():continue
        assert p.suffix=='.json' and p.stem not in active
        h=sha(p);stat=p.stat();q=dest/p.relative_to(D);q.parent.mkdir(parents=True,exist_ok=True)
        if not q.exists():shutil.copy2(p,q)
        assert sha(q)==h and sha(p)==h and p.stat().st_size==stat.st_size and p.stat().st_mtime_ns==stat.st_mtime_ns
        p.unlink();p.symlink_to(q.resolve());records.append(dict(source=str(p),ssd=str(q.resolve()),sha256=h,bytes=stat.st_size))
    dump(D/'SUPERSEDED_METADATA_RELOCATION.json',dict(files=records,bytes=sum(r['bytes'] for r in records),
        scope='superseded unexecuted preparation JSON, preserved on SSD with compatibility links; not an independent backup'))
    print('verified relocation',len(records),sum(r['bytes'] for r in records))

if __name__=='__main__':main()
