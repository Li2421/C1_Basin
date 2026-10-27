"""Opening record: frozen search code/protocol hashes, sealed parent COMPLETE hash, read-only git snapshots."""
from pathlib import Path
import sys
S2 = Path(__file__).resolve().parent
sys.path.insert(0, str(S2))
from ceiling_common import *  # noqa: F401,F403
import os
import subprocess

CODE = ('PROTOCOL.md', 'ceiling_common.py', 'ceiling_worker.py', 'ceiling_worker.sbatch', 'ceiling_replay.py', 'ceiling_driver.py', 'ceiling_start.py')
REPOS = ('/home/zhihan/research/Basin_C1', '/home/zhihan/research/02_C1_Toy_GiveWay')


def git_snapshot(repo):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')

    def run(*args):
        return subprocess.check_output(['git', '-C', repo, *args], env=env)
    status = '\n'.join(s for s in run('status', '--porcelain').decode().splitlines() if 'baseline_literature_scenes_supplement_eta_swap8_v1' not in s)
    return dict(head=run('rev-parse', 'HEAD').decode().strip(), status_sha256=hashlib.sha256(status.encode()).hexdigest(),
                diff_sha256=hashlib.sha256(run('diff', '--no-ext-diff', '--binary')).hexdigest())


def snapshot():
    return dict(code={name: sha(S2 / name) for name in CODE}, parent_complete_sha256=sha(HERE / 'COMPLETE.json'), git={r: git_snapshot(r) for r in REPOS})


if __name__ == '__main__':
    if sys.argv[1:] == ['end']:
        start = read(S2 / 'START.json'); now = snapshot()
        out = dict(utc=utc(), code_unchanged=now['code'] == start['code'], parent_complete_unchanged=now['parent_complete_sha256'] == start['parent_complete_sha256'],
                   git_unchanged={r: now['git'][r] == start['git'][r] for r in REPOS}, end=now)
        out['passed'] = out['code_unchanged'] and out['parent_complete_unchanged'] and all(out['git_unchanged'].values())
        write2(S2 / 'END.json', out); print(out['passed'], out['git_unchanged'], flush=True)
    else:
        assert not (S2 / 'START.json').exists()
        for name in FROZEN:
            verify_receipt(name)
        write2(S2 / 'START.json', dict(utc=utc(), experiment_uid=EXP2, **snapshot()))
        print('start recorded', flush=True)
