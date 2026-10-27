"""Ordered search stages; selection uses training cohort only; held-out cohort used once for final winners."""
from pathlib import Path
import sys
S2 = Path(__file__).resolve().parent
sys.path.insert(0, str(S2))
from ceiling_common import *  # noqa: F401,F403
from collections import defaultdict
from itertools import permutations
import os
import subprocess
import time
import fcntl

PY = sys.executable
CAP = 18


def jobs():
    lines = subprocess.check_output(['squeue', '-h', '-r', '-o', '%i|%u|%j|%T|%C'], text=True).splitlines()
    return [dict(zip(('id', 'user', 'name', 'state', 'cpus'), line.split('|'))) for line in lines]


def capacity():
    rows = jobs(); mine = sum(int(r['cpus']) for r in rows if r['user'] == os.environ.get('USER', 'zhihan'))
    return max(0, min(CAP - 1 - mine, 24 - 1 - sum(int(r['cpus']) for r in rows))), rows


def ours():
    return [r for r in jobs() if r['user'] == os.environ.get('USER', 'zhihan') and r['name'] == 'ceil_search']


def progress(stage, note=''):
    folder = S2 / stage / 'raw'; count = sum(1 for _ in folder.glob('*.json')) if folder.exists() else 0
    receipt = S2 / 'QUEUES' / stage / 'QUEUE.receipt.json'; planned = read(receipt)['missing'] if receipt.exists() else None
    write2(S2 / 'PROGRESS.json', dict(utc=utc(), stage=stage, new_raw=count, planned_new=planned, jobs=ours(), note=note))


def incomplete(stage):
    out = []
    for e in read(S2 / 'QUEUES' / stage / 'INDEX.json'):
        if any(not (S2 / stage / 'raw' / f"{c['rollout_uid']}.json").exists() for c in read(e['path'])):
            out.append(e)
    return out


def merge():
    with (S2 / 'MERGER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(240):
            # Another experiment's single merger may hold the shared DB; wait instead of failing.
            if subprocess.run(['pgrep', '-af', '[s]hared_rollout_db.src.merger'], capture_output=True, text=True).returncode != 0:
                break
            time.sleep(30)
        else:
            raise RuntimeError('Another merger stayed active for 2 hours')
        p = subprocess.run([PY, '-B', '-m', 'shared_rollout_db.src.merger', '--experiment', EXP2], cwd=ROOT, capture_output=True, text=True)
        write2(S2 / 'MERGES' / f'{time.time_ns()}.json', dict(utc=utc(), returncode=p.returncode, stdout=p.stdout, stderr=p.stderr[-4000:]))
        assert p.returncode == 0, p.stderr[-2000:]


def execute(stage, cases):
    marker = S2 / f'{stage}_BATCH_DONE.json'
    if marker.exists():
        return
    if not (S2 / f'{stage}_PREFLIGHT.receipt.json').exists():
        register2(stage, cases)
    if not (S2 / 'QUEUES' / stage / 'QUEUE.receipt.json').exists():
        build_queue(stage)
    (S2 / 'logs').mkdir(exist_ok=True)
    for iteration in range(6):
        missing = incomplete(stage)
        if not missing:
            merge(); integrity('after_' + stage)
            write2(marker, dict(utc=utc(), stage=stage, completed=True, unknowns_not_retried=True)); return
        assert not ours(), 'Another search batch is active'
        if iteration:
            folder = S2 / 'QUEUES' / stage / 'claims'; archive = S2 / 'QUEUES' / stage / 'aborted_claims'; archive.mkdir(exist_ok=True)
            for e in missing:
                p = folder / e['group']
                if p.exists():
                    p.rename(archive / (e['group'] + '_' + str(time.time_ns())))
        integrity('before_' + stage + '_' + str(iteration))
        while True:
            n, before = capacity()
            if n:
                break
            progress(stage, 'waiting for CPU capacity'); time.sleep(30)
        n = min(n, len(missing))
        job = subprocess.check_output(['sbatch', '--parsable', f'--array=0-{n - 1}', f'--export=ALL,CEILING_STAGE={stage}', str(S2 / 'ceiling_worker.sbatch')], text=True).strip()
        write2(S2 / 'submissions' / f'{time.time_ns()}_{stage}_{job}.json', dict(utc=utc(), job=job, workers=n, all_jobs_before=before, stage=stage))
        time.sleep(5)
        while ours():
            if (S2 / 'FATAL_STOP.json').exists():
                raise RuntimeError('Search fatal stop')
            progress(stage); time.sleep(30)
        progress(stage, 'batch ended')
    raise RuntimeError('Repeated infrastructure failures: ' + stage)


# ---------- candidate bookkeeping (training cohort only) ----------

def actions_store():
    p = S2 / 'ACTIONS.json'
    return read(p) if p.exists() else {s: {} for s in SCENES}


def remember(scene, acts):
    store = actions_store()
    for a in acts:
        old = store[scene].get(a['key'])
        assert old is None or canonical(old) == canonical(a), ('action key conflict', scene, a['key'])
        store[scene][a['key']] = a
    write2(S2 / 'ACTIONS.json', store)


def screen_pairs(scene):
    if scene == 'lit_swap8_moving':
        return []
    ch = read(HERE / 'NEW_FROZEN_CHOICES.json')['scenes'][scene]
    out = []
    for r in ch['pair_ranking']:
        a, b = map(int, r['action'].replace('eta', '').split('_to_')); out.append(act_k16(a, b))
    return out


def training_cases(scene, acts, seed=0):
    return [make(scene, ctl, s, a, seed) for ctl in train_ctls(scene) for s in train_states(scene) for a in acts]


def training_rows(stages, scene, seeds=(0,)):
    rows = []
    for st in stages:
        if (S2 / f'{st}_CASES.json').exists():
            rows += [r for r in stage_rows(st) if r['scene'] == scene and r['seed'] in seeds and r['state'] < 16]
    uniq = {}
    for r in rows:
        uniq[r['rollout_uid']] = r
    return list(uniq.values())


def rank(rows, keys, seeds=(0,)):
    by = defaultdict(list)
    for r in rows:
        if r['seed'] in seeds:
            by[(r['action'], r['type'])].append(r['success'])
    out = []
    for i, k in enumerate(keys):
        types = sorted({t for (a, t) in by if a == k}); assert len(types) == 7, (k, types)
        n = [len(by[(k, t)]) for t in types]; assert all(x == 16 * len(seeds) for x in n), (k, n)
        out.append(dict(key=k, order=i, success=float(np.mean([np.mean(by[(k, t)]) for t in types]))))
    out.sort(key=lambda r: (-r['success'], r['order']))
    return out


FIXED_STAGES = ('global', 'refine')
ALL_SEED0 = ('global', 'refine', 'schedules')


def fixed_keys(scene, include_refine):
    keys = ['safety'] + [f'eta{i}' for i in range(16)] + [f'sE{i:02d}' for i in range(64)]
    if include_refine:
        keys += [f'rf_{SHORT[scene]}_{c}_{k}' for c in range(4) for k in range(8)]
    return keys


def schedule_keys(scene):
    store = actions_store()[scene]
    return [k for k, a in store.items() if a['theta'] is not None]


def stage_global():
    cases = []
    for scene in SCENES:
        acts = [act_safety()] + k16_actions() + [act_k16(13, 10)] + sobol_actions() + screen_pairs(scene)
        remember(scene, acts); cases += training_cases(scene, acts)
    execute('global', cases)


def stage_refine():
    cases = []; centers = {}
    for idx, scene in enumerate(SCENES):
        rows = training_rows(('global',), scene); top = rank(rows, fixed_keys(scene, False))[:4]
        store = actions_store()[scene]; rng = np.random.default_rng(SEED + idx); acts = []
        for c, r in enumerate(top):
            center = np.asarray(store[r['key']]['eta_a'], float)
            for k in range(8):
                eta = np.clip(center + SIGMA * rng.standard_normal(3), BOX_E[:, 0], BOX_E[:, 1])
                acts.append(act_fixed(f'rf_{SHORT[scene]}_{c}_{k}', eta))
        centers[scene] = top; remember(scene, acts); cases += training_cases(scene, acts)
    if not (S2 / 'REFINE_CENTERS.json').exists():
        write2(S2 / 'REFINE_CENTERS.json', dict(utc=utc(), centers=centers, rule='top-4 seed-0 training success among safety, K16, 64 extended-box Sobol'))
    execute('refine', cases)


def stage_schedules():
    cases = []; chosen = {}
    for scene in SCENES:
        rows = training_rows(FIXED_STAGES, scene); top = rank(rows, fixed_keys(scene, True))[:4]
        store = actions_store()[scene]; acts = []
        for a, b in permutations([r['key'] for r in top], 2):
            for th in THETAS:
                acts.append(act_schedule(f'sc_{a}__{b}__th{int(round(th * 100)):02d}', store[a]['eta_a'], store[b]['eta_a'], th))
        chosen[scene] = top; remember(scene, acts); cases += training_cases(scene, acts)
    if not (S2 / 'SCHEDULE_COMPONENTS.json').exists():
        write2(S2 / 'SCHEDULE_COMPONENTS.json', dict(utc=utc(), components=chosen, thetas=THETAS, rule='top-4 seed-0 training success among all fixed candidates'))
    execute('schedules', cases)


def seed0_rankings(scene):
    rows = training_rows(ALL_SEED0, scene)
    return rows, rank(rows, fixed_keys(scene, True)), rank(rows, schedule_keys(scene))


def stage_recheck():
    cases = []; picks = {}
    for scene in SCENES:
        _, fixed, sched = seed0_rankings(scene); store = actions_store()[scene]
        keys = [r['key'] for r in fixed[:5]] + [r['key'] for r in sched[:5]]
        picks[scene] = dict(fixed=fixed[:5], schedules=sched[:5])
        cases += training_cases(scene, [store[k] for k in keys], seed=1)
    if not (S2 / 'RECHECK_PICKS.json').exists():
        write2(S2 / 'RECHECK_PICKS.json', dict(utc=utc(), picks=picks, rule='top-5 fixed and top-5 schedules by seed-0 training success'))
    execute('recheck', cases)


def final_choices():
    p = S2 / 'FINAL_CHOICES.json'
    if p.exists():
        return read(p)['scenes']
    out = {}
    for scene in SCENES:
        picks = read(S2 / 'RECHECK_PICKS.json')['picks'][scene]
        rows = training_rows(ALL_SEED0 + ('recheck',), scene, seeds=(0, 1)); result = {}
        for fam in ('fixed', 'schedules'):
            keys = [r['key'] for r in picks[fam]]; two = {r['key']: r['success'] for r in rank(rows, keys, seeds=(0, 1))}
            order = sorted(keys, key=lambda k: (-two[k], keys.index(k)))
            result[fam] = dict(best=order[0], two_seed=two, seed0_order=keys)
        out[scene] = result
    write2(p, dict(utc=utc(), scenes=out, selection_uses_heldout=False, rule='best two-seed training mean among top-5; ties by seed-0 order'))
    seal2('FINAL_CHOICES.receipt.json', [p], before_heldout=True)
    return out


def stage_cellcert():
    cases = []; plan = {}
    for scene in SCENES:
        rows, fixed, sched = seed0_rankings(scene); store = actions_store()[scene]
        order = {r['key']: i for i, r in enumerate(sorted(fixed + sched, key=lambda r: (-r['success'], r['order'])))}
        cells = defaultdict(list)
        for r in rows:
            if r['success'] and r['action'] in order:
                cells[(r['controller'], r['state'])].append(r['action'])
        ctl_by_name = {c['controller']: c for c in train_ctls(scene)}; st_by_index = {s['index']: s for s in train_states(scene)}
        plan[scene] = {}
        for (ctl, st), acts in sorted(cells.items()):
            best = min(acts, key=lambda k: order[k]); plan[scene][f'{ctl}|{st}'] = best
            cases.append(make(scene, ctl_by_name[ctl], st_by_index[st], store[best], 1))
    if not (S2 / 'CELL_CERT_PLAN.json').exists():
        write2(S2 / 'CELL_CERT_PLAN.json', dict(utc=utc(), plan=plan, rule='per cell: highest-ranked seed-0 success candidate rerun at seed 1'))
    execute('cellcert', cases)


def stage_heldout():
    finals = final_choices(); cases = []; arms = {}
    for scene in SCENES:
        store = actions_store()[scene]
        if scene == 'lit_swap8_moving':
            ch = read(HERE / 'MOVING_FROZEN_CHOICES.json')['choice']
        else:
            ch = read(HERE / 'NEW_FROZEN_CHOICES.json')['scenes'][scene]
        screen = [ch['best_fixed'], ch['best_pair']]
        acts = [store[finals[scene]['fixed']['best']], store[finals[scene]['schedules']['best']], act_safety(), act_k16(13), act_k16(13, 10)]
        for key in screen:
            z = key.replace('eta', '').split('_to_'); acts.append(act_k16(*map(int, z)))
        acts = list({a['key']: a for a in acts}.values()); arms[scene] = [a['key'] for a in acts]
        cases += [make(scene, ctl, s, a, 0) for ctl in qualified(scene) for s in eval_states(scene) for a in acts]
    if not (S2 / 'HELDOUT_ARMS.json').exists():
        write2(S2 / 'HELDOUT_ARMS.json', dict(utc=utc(), arms=arms, cohort='evaluation states x all qualified instances x seed 0'))
    execute('heldout', cases)


def main():
    with (S2 / 'DRIVER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write2(S2 / 'DRIVER_PID.json', dict(pid=os.getpid(), utc=utc()))
        assert read(S2 / 'REPLAY_CHECK.json')['passed']
        stage_global(); stage_refine(); stage_schedules(); stage_recheck(); final_choices(); stage_cellcert(); stage_heldout()
        write2(S2 / 'COMPUTATION_DONE.json', dict(utc=utc(), stages=['global', 'refine', 'schedules', 'recheck', 'cellcert', 'heldout']))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write2(S2 / 'DRIVER_ERROR.json', dict(utc=utc(), error=repr(exc)))
        raise
