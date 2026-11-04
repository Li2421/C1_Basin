"""Move this finished experiment's pure journal directory after verified copy.

Run only after committing/pushing this code/protocol. Never move the active
global SQLite database, WAL, SHM, or another experiment's journals.
"""
from native_io import *
import shutil

def main():
    assert Path('/mnt/ssd').is_mount();status=read(D/'SUPERVISOR_EXIT.json')
    assert status['completed']==status['total']==375 and not status['failed']
    assert not read(D/'RESOURCE_STATUS.json')['active']
    src=ROOT/'shared_rollout_db/journals'/EXP
    dst=Path('/mnt/ssd/zhihan/Basin_C1/shared_rollout_db/journals')/EXP
    if src.is_symlink():
        assert src.resolve()==dst
        certificate=read(D/'JOURNAL_STORAGE_CHECK.json');assert certificate['passed']
        for r in certificate['files']:assert sha(dst/r['name'])==r['sha256']
        print('completed journal storage already verified');return
    assert src.is_dir() and not dst.exists()
    files=sorted(src.iterdir());assert len(files)==375 and all(p.is_file() and p.suffix=='.jsonl' for p in files)
    stats={p.name:(p.stat().st_size,p.stat().st_mtime_ns,sha(p)) for p in files}
    dst.mkdir(parents=True)
    for p in files:
        shutil.copy2(p,dst/p.name);assert sha(dst/p.name)==stats[p.name][2]
    assert [p.name for p in sorted(src.iterdir())]==[p.name for p in files]
    assert all((p.stat().st_size,p.stat().st_mtime_ns,sha(p))==stats[p.name] for p in files)
    staged=src.with_name(src.name+'_verified_home_copy');assert not staged.exists()
    src.rename(staged);src.symlink_to(dst,target_is_directory=True)
    assert all(sha(src/p.name)==stats[p.name][2] and sha(staged/p.name)==stats[p.name][2] for p in files)
    shutil.rmtree(staged)
    dump(D/'JOURNAL_STORAGE_CHECK.json',dict(passed=True,source=str(src),ssd=str(dst),files=[dict(name=n,bytes=v[0],sha256=v[2]) for n,v in stats.items()],
        sqlite_moved=False,writer_processes_finished=True,independent_backup=False))
    print('completed journal storage verified',len(files))

if __name__=='__main__':main()
