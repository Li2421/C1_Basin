"""Search driver revision 3 scheduling (user 2026-10-24: results sooner): equal CPU share per active track, longest-first ~10-minute groups.
Revision 2 (user 2026-10-23): swap8 and circle tracks run concurrently; workers fill every CPU
other users do not use (one CPU kept for management) and yield when other users wait for CPUs.
Selection uses the training cohort only; the held-out cohort is used once for each track's final winners."""
from pathlib import Path
import sys
S2 = Path(__file__).resolve().parent
sys.path.insert(0, str(S2))
from ceiling_common import *  # noqa: F401,F403
from ceiling_circle import make_any, CIRCLE
from collections import defaultdict
from itertools import permutations
import math
import os
import subprocess
import time
import fcntl

PY = sys.executable
TOTAL = 24
MANAGEMENT = 1
SHORT2 = dict(SHORT, **{CIRCLE: 'circle'})
SCENE_INDEX = {'lit_swap8_free': 0, 'lit_swap8_obstacles': 1, 'lit_swap8_moving': 2, CIRCLE: 3}
TRACKS = {'swap8': dict(scenes=SCENES, prefix='', seconds=3.0), 'circle': dict(scenes=(CIRCLE,), prefix='circle_', seconds=29.0)}
KINDS = ('global', 'refine', 'schedules', 'recheck', 'cellcert', 'heldout')
LEGACY = {'ceil_search': 'global'}


def me():
    return os.environ.get('USER', 'zhihan')


def stage_name(track, kind):
    return TRACKS[track]['prefix'] + kind


def worker_stage(name):
    if name in LEGACY:
        return LEGACY[name]
    return name[3:] if name.startswith('cs_') else None


def squeue():
    lines = subprocess.check_output(['squeue', '-h', '-r', '-o', '%i|%u|%j|%T|%C|%r'], text=True).splitlines()
    return [dict(zip(('id', 'user', 'name', 'state', 'cpus', 'reason'), line.split('|'))) for line in lines]


# ---------- candidate bookkeeping (training cohort only) ----------

def actions_store():
    p = S2 / 'ACTIONS.json'
    store = read(p) if p.exists() else {}
    for s in SCENES + (CIRCLE,):
        store.setdefault(s, {})
    return store


def remember(scene, acts):
    store = actions_store()
    for a in acts:
        old = store[scene].get(a['key'])
        assert old is None or canonical(old) == canonical(a), ('action key conflict', scene, a['key'])
        store[scene][a['key']] = a
    write2(S2 / 'ACTIONS.json', store)


def frozen_choice(scene):
    if scene == CIRCLE:
        return read(HERE / 'OLD_FROZEN_CHOICES.json')['scenes'][scene]
    if scene == 'lit_swap8_moving':
        return read(HERE / 'MOVING_FROZEN_CHOICES.json')['choice']
    return read(HERE / 'NEW_FROZEN_CHOICES.json')['scenes'][scene]


def screen_pairs(scene):
    if scene == 'lit_swap8_moving':
        return []
    out = []
    for r in frozen_choice(scene)['pair_ranking']:
        a, b = map(int, r['action'].replace('eta', '').split('_to_')); out.append(act_k16(a, b))
    return out


def training_cases(scene, acts, seed=0):
    return [make_any(scene, ctl, s, a, seed) for ctl in train_ctls(scene) for s in train_states(scene) for a in acts]


def training_rows(stages, scene, seeds=(0,)):
    train = {s['index'] for s in train_states(scene)}; rows = {}
    for st in stages:
        if (S2 / f'{st}_CASES.json').exists():
            for r in stage_rows(st):
                if r['scene'] == scene and r['seed'] in seeds and r['state'] in train:
                    rows[r['rollout_uid']] = r
    return list(rows.values())


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


def fixed_keys(scene, include_refine):
    keys = ['safety'] + [f'eta{i}' for i in range(16)] + [f'sE{i:02d}' for i in range(64)]
    if include_refine:
        keys += [f'rf_{SHORT2[scene]}_{c}_{k}' for c in range(4) for k in range(8)]
    return keys


def schedule_keys(scene):
    return [k for k, a in actions_store()[scene].items() if a['theta'] is not None]


def seed0_stages(track):
    return tuple(stage_name(track, k) for k in ('global', 'refine', 'schedules'))


def build_global(track):
    cases = []
    for scene in TRACKS[track]['scenes']:
        acts = [act_safety()] + k16_actions() + [act_k16(13, 10)] + sobol_actions() + screen_pairs(scene)
        remember(scene, acts); cases += training_cases(scene, acts)
    return cases


def build_refine(track):
    cases = []; centers = {}
    for scene in TRACKS[track]['scenes']:
        rows = training_rows((stage_name(track, 'global'),), scene); top = rank(rows, fixed_keys(scene, False))[:4]
        store = actions_store()[scene]; rng = np.random.default_rng(SEED + SCENE_INDEX[scene]); acts = []
        for c, r in enumerate(top):
            center = np.asarray(store[r['key']]['eta_a'], float)
            for k in range(8):
                eta = np.clip(center + SIGMA * rng.standard_normal(3), BOX_E[:, 0], BOX_E[:, 1])
                acts.append(act_fixed(f'rf_{SHORT2[scene]}_{c}_{k}', eta))
        centers[scene] = top; remember(scene, acts); cases += training_cases(scene, acts)
    write2(S2 / f'REFINE_CENTERS_{track}.json', dict(utc=utc(), centers=centers, rule='top-4 seed-0 training success among safety, history 16, 64 extended-box Sobol'))
    return cases


def build_schedules(track):
    cases = []; chosen = {}
    for scene in TRACKS[track]['scenes']:
        rows = training_rows(tuple(stage_name(track, k) for k in ('global', 'refine')), scene); top = rank(rows, fixed_keys(scene, True))[:4]
        store = actions_store()[scene]; acts = []
        for a, b in permutations([r['key'] for r in top], 2):
            for th in THETAS:
                acts.append(act_schedule(f'sc_{a}__{b}__th{int(round(th * 100)):02d}', store[a]['eta_a'], store[b]['eta_a'], th))
        chosen[scene] = top; remember(scene, acts); cases += training_cases(scene, acts)
    write2(S2 / f'SCHEDULE_COMPONENTS_{track}.json', dict(utc=utc(), components=chosen, thetas=THETAS, rule='top-4 seed-0 training success among all fixed candidates'))
    return cases


def seed0_rankings(track, scene):
    rows = training_rows(seed0_stages(track), scene)
    return rows, rank(rows, fixed_keys(scene, True)), rank(rows, schedule_keys(scene))


def build_recheck(track):
    cases = []; picks = {}
    for scene in TRACKS[track]['scenes']:
        _, fixed, sched = seed0_rankings(track, scene); store = actions_store()[scene]
        keys = [r['key'] for r in fixed[:5]] + [r['key'] for r in sched[:5]]
        picks[scene] = dict(fixed=fixed[:5], schedules=sched[:5])
        cases += training_cases(scene, [store[k] for k in keys], seed=1)
    write2(S2 / f'RECHECK_PICKS_{track}.json', dict(utc=utc(), picks=picks, rule='top-5 fixed and top-5 schedules by seed-0 training success'))
    return cases


def final_choices(track):
    p = S2 / f'FINAL_CHOICES_{track}.json'
    if p.exists():
        return read(p)['scenes']
    out = {}
    for scene in TRACKS[track]['scenes']:
        picks = read(S2 / f'RECHECK_PICKS_{track}.json')['picks'][scene]
        rows = training_rows(seed0_stages(track) + (stage_name(track, 'recheck'),), scene, seeds=(0, 1)); result = {}
        for fam in ('fixed', 'schedules'):
            keys = [r['key'] for r in picks[fam]]; two = {r['key']: r['success'] for r in rank(rows, keys, seeds=(0, 1))}
            order = sorted(keys, key=lambda k: (-two[k], keys.index(k)))
            result[fam] = dict(best=order[0], two_seed=two, seed0_order=keys)
        out[scene] = result
    write2(p, dict(utc=utc(), scenes=out, selection_uses_heldout=False, rule='best two-seed training mean among top-5; ties by seed-0 order'))
    seal2(f'FINAL_CHOICES_{track}.receipt.json', [p], before_heldout=True)
    return out


def build_cellcert(track):
    cases = []; plan = {}
    for scene in TRACKS[track]['scenes']:
        rows, fixed, sched = seed0_rankings(track, scene); store = actions_store()[scene]
        order = {r['key']: i for i, r in enumerate(sorted(fixed + sched, key=lambda r: (-r['success'], r['order'])))}
        cells = defaultdict(list)
        for r in rows:
            if r['success'] and r['action'] in order:
                cells[(r['controller'], r['state'])].append(r['action'])
        ctl_by_name = {c['controller']: c for c in train_ctls(scene)}; st_by_index = {s['index']: s for s in train_states(scene)}
        plan[scene] = {}
        for (ctl, st), acts in sorted(cells.items()):
            best = min(acts, key=lambda k: order[k]); plan[scene][f'{ctl}|{st}'] = best
            cases.append(make_any(scene, ctl_by_name[ctl], st_by_index[st], store[best], 1))
    write2(S2 / f'CELL_CERT_PLAN_{track}.json', dict(utc=utc(), plan=plan, rule='per cell: highest-ranked seed-0 success candidate rerun at seed 1'))
    return cases


def build_heldout(track):
    finals = final_choices(track); cases = []; arms = {}
    for scene in TRACKS[track]['scenes']:
        store = actions_store()[scene]; ch = frozen_choice(scene)
        acts = [store[finals[scene]['fixed']['best']], store[finals[scene]['schedules']['best']], act_safety(), act_k16(13), act_k16(13, 10)]
        for key in (ch['best_fixed'], ch['best_pair']):
            z = key.replace('eta', '').split('_to_'); acts.append(act_k16(*map(int, z)))
        acts = list({a['key']: a for a in acts}.values()); arms[scene] = [a['key'] for a in acts]
        cases += [make_any(scene, ctl, s, a, 0) for ctl in qualified(scene) for s in eval_states(scene) for a in acts]
    write2(S2 / f'HELDOUT_ARMS_{track}.json', dict(utc=utc(), arms=arms, cohort='evaluation states x all qualified instances x seed 0'))
    return cases


BUILDERS = dict(global_=build_global, refine=build_refine, schedules=build_schedules, recheck=build_recheck, cellcert=build_cellcert, heldout=build_heldout)


# ---------- stage lifecycle ----------

def current_stage(track):
    if track == 'circle' and not read(S2 / 'REPLAY_CHECK_CIRCLE.json')['passed']:
        raise RuntimeError('circle replay check failed')
    for kind in KINDS:
        stage = stage_name(track, kind)
        if (S2 / f'{stage}_BATCH_DONE.json').exists():
            continue
        if not (S2 / f'{stage}_PREFLIGHT.receipt.json').exists():
            register2(stage, BUILDERS['global_' if kind == 'global' else kind](track))
        if not (S2 / 'QUEUES' / stage / 'QUEUE.receipt.json').exists():
            build_queue2(stage, track)
        return stage
    return None


def expected_seconds(track):
    """Mean measured seconds per (scene, controller type) from the track's completed global stage."""
    by = defaultdict(list)
    for p in (S2 / stage_name(track, 'global') / 'raw').glob('*.json'):
        r = read(p); by[(r['case']['state']['scene'], r['case']['controller_payload']['nominal']['type_code'])].append(r['elapsed_seconds'])
    return {k: float(np.mean(v)) for k, v in by.items()}


def build_queue2(stage, track, target=600.):
    """Revision 3 scheduling only: longest expected groups first, groups of about ten minutes; cases and identities unchanged."""
    folder = S2 / 'QUEUES' / stage
    assert not (folder / 'QUEUE.receipt.json').exists(), stage
    todo = read(S2 / f'{stage}_MISSING_CASES.json'); seconds = expected_seconds(track); default = TRACKS[track]['seconds']; cells = {}
    with dbm.connect(True) as db:
        for c in todo:
            assert db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?', (c['rollout_uid'],)).fetchone() is None, c['rollout_uid']
            s = c['state']; key = (s['scene'], c['controller_payload']['nominal']['controller'], s['N'], s['index'], c['seed'])
            cells.setdefault(key, []).append(c)
    index = []
    for key, rows in cells.items():
        rows.sort(key=lambda x: x['requested_action'])
        each = seconds.get((key[0], rows[0]['controller_payload']['nominal']['type_code']), default)
        chunk = max(1, min(16, int(target // each)))
        for k in range(0, len(rows), chunk):
            part = rows[k:k + chunk]; gid = uid('group', [list(key), k])
            p = folder / 'groups' / f'{gid}.json'; write2(p, part)
            index.append(dict(group=gid, path=str(p), sha256=sha(p), count=len(part), expected_seconds=len(part) * each))
    index.sort(key=lambda r: (-r['expected_seconds'], r['group']))
    write2(folder / 'INDEX.json', index)
    write2(folder / 'QUEUE.receipt.json', dict(utc=utc(), stage=stage, missing=sum(r['count'] for r in index), groups=len(index), order='longest expected first',
                                               files={str(p): sha(p) for p in (S2 / f'{stage}_MISSING_CASES.json', folder / 'INDEX.json')}))
    with (S2 / 'REQUEST_LEDGER.jsonl').open('a') as f:
        f.write(canonical(read(folder / 'QUEUE.receipt.json')) + '\n')
    print(stage, 'groups', len(index), 'missing', sum(r['count'] for r in index), flush=True)


_INDEX = {}


def index_of(stage):
    if stage not in _INDEX:
        _INDEX[stage] = read(S2 / 'QUEUES' / stage / 'INDEX.json')
    return _INDEX[stage]


def status(stage, rows):
    claims = S2 / 'QUEUES' / stage / 'claims'
    claimed = set(os.listdir(claims)) if claims.exists() else set()
    complete = {g for g in claimed if (claims / g / 'COMPLETE.json').exists()}
    groups = index_of(stage)
    workers = [r for r in rows if r['user'] == me() and worker_stage(r['name']) == stage]
    return dict(groups=len(groups), unclaimed=[e for e in groups if e['group'] not in claimed], complete=len(complete),
                in_progress=sorted(claimed - complete), workers=workers)


def retries():
    p = S2 / 'RETRIES.json'
    return read(p) if p.exists() else {}


def unstick(stage, st):
    count = retries(); count[stage] = count.get(stage, 0) + 1
    if count[stage] > 5:
        raise RuntimeError('Repeated infrastructure failures: ' + stage)
    folder = S2 / 'QUEUES' / stage; archive = folder / 'aborted_claims'; archive.mkdir(exist_ok=True)
    for g in st['in_progress']:
        (folder / 'claims' / g).rename(archive / f'{g}_{time.time_ns()}')
    write2(S2 / 'RETRIES.json', count)


def finalize(stage):
    todo = read(S2 / f'{stage}_MISSING_CASES.json')
    lacking = [c['rollout_uid'] for c in todo if not (S2 / stage / 'raw' / f"{c['rollout_uid']}.json").exists()]
    assert not lacking, (stage, len(lacking))
    merge(); integrity('after_' + stage)
    write2(S2 / f'{stage}_BATCH_DONE.json', dict(utc=utc(), stage=stage, completed=True, unknowns_not_retried=True))


def merge():
    with (S2 / 'MERGER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(240):
            if subprocess.run(['pgrep', '-af', '[s]hared_rollout_db.src.merger'], capture_output=True, text=True).returncode != 0:
                break
            time.sleep(30)
        else:
            raise RuntimeError('Another merger stayed active for 2 hours')
        p = subprocess.run([PY, '-B', '-m', 'shared_rollout_db.src.merger', '--experiment', EXP2], cwd=ROOT, capture_output=True, text=True)
        write2(S2 / 'MERGES' / f'{time.time_ns()}.json', dict(utc=utc(), returncode=p.returncode, stdout=p.stdout, stderr=p.stderr[-4000:]))
        assert p.returncode == 0, p.stderr[-2000:]


# ---------- CPU use: everything other users do not use, minus one management CPU ----------

def capacity(rows):
    allocated = sum(int(r['cpus']) for r in rows if r['state'] == 'RUNNING')
    mine_queued = sum(int(r['cpus']) for r in rows if r['user'] == me() and r['state'] == 'PENDING')
    others_waiting = sum(int(r['cpus']) for r in rows if r['user'] != me() and r['state'] == 'PENDING' and r['reason'] in ('Resources', 'Priority'))
    idle = TOTAL - allocated
    free = idle - MANAGEMENT - mine_queued
    return free, max(0, others_waiting - max(0, idle - MANAGEMENT)), idle


def yield_cpus(rows, need, signaled):
    log = []
    for r in rows:
        if r['user'] == me() and worker_stage(r['name']) and r['state'] == 'PENDING':
            subprocess.run(['scancel', r['id']], check=False); log.append(('cancel_pending', r['id']))
    running = sorted((r for r in rows if r['user'] == me() and worker_stage(r['name']) and r['state'] == 'RUNNING' and r['id'] not in signaled),
                     key=lambda r: [int(x) for x in r['id'].replace('_', '.').split('.') if x.isdigit()], reverse=True)
    for r in running[:need]:
        subprocess.run(['scancel', '--batch', '--signal=USR1', r['id']], check=False); signaled.add(r['id']); log.append(('yield', r['id']))
    if log:
        with (S2 / 'YIELD_LOG.jsonl').open('a') as f:
            f.write(canonical(dict(utc=utc(), need=need, actions=log)) + '\n')


def track_of(stage):
    return 'circle' if stage.startswith('circle_') else 'swap8'


def allocate(rows, active, free, state):
    """Equal share of my usable CPUs per active track (capped by its remaining groups); submit into free CPUs,
    otherwise move workers from a track above its share by a graceful yield (at most every five minutes)."""
    workers = defaultdict(list)
    for r in rows:
        if r['user'] == me() and worker_stage(r['name']):
            workers[track_of(worker_stage(r['name']))].append(r)
    tracks = {track_of(stage): (stage, st) for stage, (track, st) in active.items()}
    if not tracks:
        return
    caps = {t: len(st['unclaimed']) + len(workers[t]) for t, (stage, st) in tracks.items()}
    pool = max(free, 0) + sum(len(workers[t]) for t in tracks); share = {t: 0 for t in tracks}; left = pool
    while left > 0 and any(share[t] < caps[t] for t in tracks):
        for t in sorted(tracks, key=lambda t: share[t]):
            if left > 0 and share[t] < caps[t]:
                share[t] += 1; left -= 1
    deficit = {t: share[t] - len(workers[t]) for t in tracks}
    if free > 0:
        for t in sorted(tracks, key=lambda t: -deficit[t]):
            n = min(deficit[t], free)
            if n <= 0:
                continue
            stage = tracks[t][0]
            job = subprocess.check_output(['sbatch', '--parsable', f'--job-name=cs_{stage}', f'--array=0-{n - 1}', f'--export=ALL,CEILING_STAGE={stage}',
                                           str(S2 / 'ceiling_worker.sbatch')], text=True).strip()
            write2(S2 / 'submissions' / f'{time.time_ns()}_{stage}_{job}.json', dict(utc=utc(), job=job, workers=n, stage=stage, free_before=free, share=share))
            free -= n
        return
    need = {t: d for t, d in deficit.items() if d >= 2}
    if not need or time.time() - state.get('last_rebalance', 0) < 300:
        return
    for t, d in need.items():
        donors = [u for u in tracks if u != t and -deficit[u] >= 1]
        for u in donors:
            k = min(d, -deficit[u])
            running = sorted((r for r in workers[u] if r['state'] == 'RUNNING' and r['id'] not in state['signaled']),
                             key=lambda r: [int(x) for x in r['id'].replace('_', '.').split('.') if x.isdigit()], reverse=True)[:k]
            for r in running:
                subprocess.run(['scancel', '--batch', '--signal=USR1', r['id']], check=False); state['signaled'].add(r['id'])
            if running:
                with (S2 / 'REBALANCE_LOG.jsonl').open('a') as f:
                    f.write(canonical(dict(utc=utc(), to=t, from_=u, workers=[r['id'] for r in running], share=share)) + '\n')
                state['last_rebalance'] = time.time()


def main():
    with (S2 / 'DRIVER2.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write2(S2 / 'DRIVER2_PID.json', dict(pid=os.getpid(), utc=utc()))
        assert read(S2 / 'REPLAY_CHECK.json')['passed']
        signaled = set(); state = dict(signaled=signaled)
        while True:
            if (S2 / 'FATAL_STOP.json').exists():
                raise RuntimeError('Search fatal stop')
            rows = squeue(); active = {}; report = {}; finished = 0
            for track in TRACKS:
                stage = current_stage(track)
                if stage is None:
                    finished += 1; report[track] = 'done'; continue
                st = status(stage, rows)
                report[track] = dict(stage=stage, groups=st['groups'], complete=st['complete'], unclaimed=len(st['unclaimed']), in_progress=len(st['in_progress']), workers=len(st['workers']))
                if st['complete'] == st['groups'] and not st['workers']:
                    finalize(stage); continue
                if not st['unclaimed'] and not st['workers'] and st['in_progress']:
                    unstick(stage, st); continue
                if st['unclaimed']:
                    active[stage] = (track, st)
            if finished == len(TRACKS):
                break
            free, need, idle = capacity(rows)
            if need > 0:
                yield_cpus(rows, need, signaled)
            elif active:
                allocate(rows, active, free, state)
            write2(S2 / 'PROGRESS2.json', dict(utc=utc(), tracks=report, free_cpus=free, idle_cpus=idle, others_waiting_cpus=need,
                                               my_workers=sum(1 for r in rows if r['user'] == me() and worker_stage(r['name']))))
            time.sleep(30)
        write2(S2 / 'COMPUTATION_DONE.json', dict(utc=utc(), tracks=list(TRACKS), stages=KINDS))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write2(S2 / 'DRIVER2_ERROR.json', dict(utc=utc(), error=repr(exc)))
        raise
