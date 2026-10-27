"""Evidence-only analysis of the ceiling search (both tracks); no rollouts, no selection on the held-out cohort."""
from pathlib import Path
import sys
S2 = Path(__file__).resolve().parent
sys.path.insert(0, str(S2))
from ceiling_common import *  # noqa: F401,F403
from ceiling_driver2 import training_rows, fixed_keys, schedule_keys, actions_store, rank, seed0_stages, stage_name, TRACKS, frozen_choice
from statistics_supplement_user_fix_20261023 import infer
from collections import defaultdict

NAMES = {'lit_swap8_free': '八机无障碍', 'lit_swap8_obstacles': '八机静态障碍', 'lit_swap8_moving': '八机移动障碍', 'lit_circle_five': '圆形五机'}


def cube(rows, keys):
    states = sorted({r['state'] for r in rows}); ctls = sorted({r['controller'] for r in rows})
    si = {s: i for i, s in enumerate(states)}; ci = {c: i for i, c in enumerate(ctls)}; ai = {a: i for i, a in enumerate(keys)}
    S = np.full((len(states), len(ctls), len(keys)), np.nan)
    for r in rows:
        if r['action'] in ai and r['seed'] == 0:
            S[si[r['state']], ci[r['controller']], ai[r['action']]] = r['success']
    assert not np.isnan(S).any(), 'incomplete training cube'
    return S, states, ctls


def tie_mean(scores, values):
    return float(values[np.isclose(scores, scores.max())].mean())


def ceiling(S):
    ns, nc, _ = S.shape
    return dict(best_single=float(S.mean(axis=(0, 1)).max()), per_state_oracle=float(S.mean(axis=1).max(axis=1).mean()),
                per_controller_oracle=float(S.mean(axis=0).max(axis=1).mean()), per_cell_oracle=float(S.max(axis=2).mean()),
                never_solved=float((S.max(axis=2) == 0).mean()),
                crossfit_fixed=float(np.mean([tie_mean(np.delete(S, s, axis=0).mean(axis=(0, 1)), S[s, c]) for s in range(ns) for c in range(nc)])),
                crossfit_state=float(np.mean([tie_mean(np.delete(S[s], c, axis=0).mean(axis=0), S[s, c]) for s in range(ns) for c in range(nc)])),
                crossfit_controller=float(np.mean([tie_mean(np.delete(S[:, c], s, axis=0).mean(axis=0), S[s, c]) for s in range(ns) for c in range(nc)])),
                cells=int(ns * nc), candidates=int(S.shape[2]))


def where(eta):
    eta = np.asarray(eta, float)
    inside_h = bool(np.all(eta >= BOX_H[:, 0] - 1e-12) and np.all(eta <= BOX_H[:, 1] + 1e-12))
    width = BOX_E[:, 1] - BOX_E[:, 0]
    near_edge = [i for i in range(3) if min(eta[i] - BOX_E[i, 0], BOX_E[i, 1] - eta[i]) <= .05 * width[i]]
    return dict(inside_history_box=inside_h, near_extended_edge=near_edge)


def counts(rows):
    return dict(n=len(rows), success=sum(r['success'] for r in rows), timeout=sum(r['timeout'] for r in rows), unknown=sum(r['unknown'] for r in rows),
                collision=sum(r['collision'] for r in rows), agent=sum(int(r['collision_types'].get('agent', 0)) for r in rows),
                static=sum(int(r['collision_types'].get('static', 0)) for r in rows), moving=sum(int(r['collision_types'].get('moving', 0)) for r in rows),
                restoration_episodes=sum(int(r['restoration_steps'] > 0) for r in rows), violation_episodes=sum(int(r['constraint_violation_steps'] > 0) for r in rows))


def type_equal(rows):
    by = defaultdict(list)
    for r in rows:
        by[r['type']].append(r['success'])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def scene_result(track, scene):
    store = actions_store()[scene]
    rows0 = training_rows(seed0_stages(track), scene)
    fixed = fixed_keys(scene, True); sched = schedule_keys(scene)
    rank_fixed = rank(rows0, fixed); rank_sched = rank(rows0, sched)
    finals = read(S2 / f'FINAL_CHOICES_{track}.json')['scenes'][scene]
    rows01 = training_rows(seed0_stages(track) + (stage_name(track, 'recheck'), stage_name(track, 'cellcert')), scene, seeds=(0, 1))
    ch = frozen_choice(scene)
    refs = list(dict.fromkeys(['safety', 'eta13', 'eta13_to_10', ch['best_fixed'], ch['best_pair'], finals['fixed']['best'], finals['schedules']['best']]))
    seed0 = {r['key']: r['success'] for r in rank_fixed + rank_sched}
    seed1 = {}
    for k in refs:
        rr = [r for r in rows01 if r['action'] == k and r['seed'] == 1]
        if len(rr) == 112:
            seed1[k] = type_equal(rr)
    screen_keys = [f'eta{i}' for i in range(16)] + [k for k in sched if k.startswith('eta') and k != 'eta13_to_10']
    keys_all = fixed + sched
    S, states, ctls = cube(rows0, keys_all)
    out = dict(scene=scene, track=track,
               training_reference={k: dict(seed0=seed0.get(k), seed1=seed1.get(k), eta_a=store[k]['eta_a'], eta_b=store[k]['eta_b'], theta=store[k]['theta']) for k in refs},
               top_fixed=[dict(r, eta=store[r['key']]['eta_a'], **where(store[r['key']]['eta_a'])) for r in rank_fixed[:10]],
               top_schedules=[dict(r, eta_a=store[r['key']]['eta_a'], eta_b=store[r['key']]['eta_b'], theta=store[r['key']]['theta']) for r in rank_sched[:10]],
               family_best={fam: max((r for r in rank_fixed if pred(r['key'])), key=lambda r: r['success']) for fam, pred in
                            (('safety', lambda k: k == 'safety'), ('history_box', lambda k: k.startswith('eta')), ('extended_box', lambda k: k.startswith('sE')),
                             ('local_refinement', lambda k: k.startswith('rf_')))},
               ceiling_all=ceiling(S), ceiling_screen_only=ceiling(S[:, :, [keys_all.index(k) for k in screen_keys]]), final_choices=finals)
    plan = read(S2 / f'CELL_CERT_PLAN_{track}.json')['plan'][scene]
    lookup = {(r['controller'], r['state'], r['action']): r['success'] for r in rows01 if r['seed'] == 1}
    certified = sum(lookup[(cell.split('|')[0], int(cell.split('|')[1]), a)] for cell, a in plan.items())
    out['certified_per_cell'] = dict(cells=len(states) * len(ctls), seed0_solvable=len(plan), seed1_reproduced=int(certified), rate=certified / (len(states) * len(ctls)))
    arms = read(S2 / f'HELDOUT_ARMS_{track}.json')['arms'][scene]
    held = [r for r in stage_rows(stage_name(track, 'heldout')) if r['scene'] == scene]
    out['heldout'] = dict(arms=arms, inference=infer(held, arms, (0,)), counts={a: counts([r for r in held if r['action'] == a]) for a in arms})
    return out


def pct(x):
    return '—' if x is None else f'{100 * x:.1f}%'


def eta_text(a):
    f = lambda e: '(' + ', '.join(f'{x:.2f}' for x in e) + ')'
    return f(a['eta_a']) if a['theta'] is None else f"{f(a['eta_a'])}→{f(a['eta_b'])}，进度阈值 {a['theta']:g}"


def report(results):
    lines = ['# 修正架构上限搜索结果', '', '训练集只用于选择；留出集只跑一次最终赢家。成功率按控制器类型等权；η 依次为（朝目标，侧移，互斥）。', '']
    for scene, r in results.items():
        store = actions_store()[scene]; ca = r['ceiling_all']; cs = r['ceiling_screen_only']; cc = r['certified_per_cell']
        lines += [f"## {NAMES[scene]}", '', '### 训练集', '', '| 做法 | 种子 0 | 种子 1 | η |', '|---|---:|---:|---|']
        for k, v in r['training_reference'].items():
            lines.append(f"| {k} | {pct(v['seed0'])} | {pct(v['seed1'])} | {eta_text(store[k])} |")
        lines += ['', '| 上限型指标（种子 0） | 原 22 个候选 | 全部候选 |', '|---|---:|---:|']
        for key, name in (('best_single', '最佳单一修正'), ('crossfit_fixed', '单一修正（交叉验证）'), ('crossfit_state', '按状态挑（交叉验证）'),
                          ('crossfit_controller', '按控制器挑（交叉验证）'), ('per_state_oracle', '逐状态最优'), ('per_controller_oracle', '逐控制器最优'),
                          ('per_cell_oracle', '逐单元最优'), ('never_solved', '所有候选都失败的单元')):
            lines.append(f'| {name} | {pct(cs[key])} | {pct(ca[key])} |')
        lines += ['', f"候选数：原 {cs['candidates']} 个，全部 {ca['candidates']} 个。逐单元最优换种子 1 复现：{cc['seed1_reproduced']}/{cc['cells']}（{pct(cc['rate'])}）。", '']
        lines += ['前 5 个固定修正：', '']
        for t in r['top_fixed'][:5]:
            lines.append(f"- {t['key']} {pct(t['success'])} {eta_text(store[t['key']])}{'' if t['inside_history_box'] else '，在历史范围外'}{'，靠近扩展范围边界' if t['near_extended_edge'] else ''}")
        lines += ['', '前 5 个切换修正：', '']
        for t in r['top_schedules'][:5]:
            lines.append(f"- {t['key']} {pct(t['success'])} {eta_text(store[t['key']])}")
        inf = r['heldout']['inference']
        lines += ['', '### 留出集', '', '| 做法 | 成功率 [95%] | 相对 η13 | 相对仅安全过滤 | 碰撞局（机器人/静态/移动） | unknown |', '|---|---|---|---|---|---:|']
        for a in r['heldout']['arms']:
            m = inf['methods'][a]; c = r['heldout']['counts'][a]
            d13 = m['versus']['eta13']; ds = m['versus']['safety']
            lines.append(f"| {a} | {pct(m['success_lower'])} [{100 * m['ci95'][0]:.1f}, {100 * m['ci95'][1]:.1f}] | {100 * d13['delta']:+.1f} [{100 * d13['ci95'][0]:.1f}, {100 * d13['ci95'][1]:.1f}] | "
                         f"{100 * ds['delta']:+.1f} [{100 * ds['ci95'][0]:.1f}, {100 * ds['ci95'][1]:.1f}] | {c['collision']}（{c['agent']}/{c['static']}/{c['moving']}） | {c['unknown']} |")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    results = {}
    for track, spec in TRACKS.items():
        for scene in spec['scenes']:
            results[scene] = scene_result(track, scene)
    write2(S2 / 'CEILING_RESULTS.json', dict(utc=utc(), scenes=results, analysis_sha256=sha(Path(__file__)), new_rollouts=0))
    (S2 / 'CEILING_REPORT.md').write_text(report(results))
    print('analysis written', flush=True)


if __name__ == '__main__':
    main()
