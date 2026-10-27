"""All 32 frozen evaluation scenes (8 validation + 12 test + 12 second-batch test), states 8-23, plain-language names.
Groups: classical (7 capable controllers) and learned (6 instances).  Prints (1) second-batch per-scene table,
(2) per-family averages over 8 scenes, (3) unweighted and type-balanced totals with stratified bootstrap."""
import os, sys, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
Z = json.load(open(os.path.join(ROOT, 'FINAL_FROZEN.json'))); F2 = json.load(open(os.path.join(ROOT, 'FROZEN_T2.json')))
VAL, TEST, T2 = Z['val'], Z['test'], F2['test2']; V = VAL + TEST + T2
CH = {**Z['choices'], **F2['choices']}
S = {s['name']: s for s in json.load(open(os.path.join(ROOT, 'scenes_all2.json')))}
norm = lambda k: k[5:] if k.startswith('menu:') else k
R = {}
for f in ['test_menu.jsonl', 'test_props.jsonl', 'test_nominal.jsonl', 'test8.jsonl', 'test_rhr.jsonl', 'ens_val.jsonl', 'fresh.jsonl', 'val8.jsonl', 'rhr_goal.jsonl', 'val_nominal.jsonl']:
    for d in map(json.loads, open(os.path.join(ROOT, f))):
        if d['scene'] in V and 8 <= d['state'] <= 23:
            R[(d['scene'], norm(d.get('key') or d['action']), d['controller'], d['state'])] = d['success']
for f in ['eval.jsonl', 't2_eval.jsonl']:
    for d in map(json.loads, open(os.path.join(HERE, f))):
        if 'robot' not in d:
            R[(d['scene'], norm(d['key']), d['controller'], d['state'])] = d['success']
M = ['pipe', 'sE22', 'rhr', 'rhr_goal', 'safety']
MN = ['流水线', '最好固定 η', '右手规则·绕中心', '右手规则·按目标方向', '只加安全投影']
key_of = lambda v, m: f"p{CH[v]['ens16']}" if m == 'pipe' else m
CL, LE = [0, 1, 2, 4, 5, 6, 7], list(range(8, 14))
assert all((v, key_of(v, m), c, s) in R for v in V for m in M + ['nominal'] for c in CL + LE for s in range(8, 24)), 'missing records'
assert all(R[(v, 'nominal', c, s)] == 0 for v in V for c in CL + LE for s in range(8, 24))
FAM = ['box', 'plus', 'obst', 'swap']
FN = {'box': '拥挤方形场地', 'plus': '十字路口', 'obst': '带障碍的圆圈对穿', 'swap': '无障碍圆圈对穿'}
byfam = collections.defaultdict(list)
for v in V: byfam[S[v]['cluster']].append(v)


def nice(v):
    p = S[v]['params']; f = S[v]['cluster']; tag = '验证' if v in VAL else '测试' if v in TEST else '第二批测试'; paper = ''
    if f == 'box': name = f"{p['N']} 台机器人，半径 {p['r']:.2f} m"
    elif f == 'plus': name = f"路宽 {2 * p['w']:.2f} m" + ('，窄' if 2 * p['w'] <= 0.5 + 1e-9 else '')
    elif f == 'obst':
        name = f"起点圆半径 {p['R']:.2f} m，机器人半径 {p['r']:g} m"; paper = '，论文原设置' if abs(p['R'] - 1.4) < 1e-9 and abs(p['r'] - 0.05) < 1e-9 else ''
    else:
        name = f"{p['N']} 台机器人，圆半径 {p['R']:g} m"; paper = '，论文原设置' if p['N'] == 8 else ''
    return f'{name}（{tag}{paper}）'


def r(vs, m, cs): return 100 * np.mean([R[(v, key_of(v, m), c, s)] for v in vs for c in cs for s in range(8, 24)])


G2 = [('经典', CL), ('学习型', LE)]
out = ['### 第二批 12 个测试场景（逐场景）']
for f in FAM:
    out.append(f'\n**{FN[f]}**\n| 场景 | 组 | ' + ' | '.join(MN) + ' |\n|---|---|' + '---:|' * len(M))
    for v in [x for x in byfam[f] if x in T2]:
        for i, (g, cs) in enumerate(G2):
            out.append(f"| {nice(v) if i == 0 else ''} | {g} | " + ' | '.join(f'{r([v], m, cs):.1f}' for m in M) + ' |')
out.append('\n### 按场景族平均（每族 8 个场景：原来 5 个 + 第二批 3 个）\n| 场景族 | 组 | ' + ' | '.join(MN) + ' |\n|---|---|' + '---:|' * len(M))
for f in FAM:
    for i, (g, cs) in enumerate(G2):
        out.append(f"| {FN[f] if i == 0 else ''} | {g} | " + ' | '.join(f'{r(byfam[f], m, cs):.1f}' for m in M) + ' |')
A = {(m, g): {v: np.array([np.mean([R[(v, key_of(v, m), c, s)] for c in cs]) for s in range(8, 24)]) for v in V} for m in M for g, cs in G2 + [('全部', CL + LE)]}


def agg(m, g, draws, balanced):
    fm = {f: np.mean([A[(m, g)][v][ss].mean() for v, ss in draws[f]]) for f in FAM}
    if balanced:
        return 100 * (0.5 * fm['box'] + 0.5 * np.mean([fm['plus'], fm['obst'], fm['swap']]))
    n = {f: len(draws[f]) for f in FAM}
    return 100 * sum(fm[f] * n[f] for f in FAM) / sum(n.values())


rng = np.random.default_rng(0)
full = {f: [(v, np.arange(16)) for v in byfam[f]] for f in FAM}
for balanced, title in [(False, '32 个场景直接平均（对心对穿占 3/4）'), (True, '两大类等权（拥挤方形场地 50%，三种对心对穿合计 50%）')]:
    out.append(f'\n### {title}\n| 组 | ' + ' | '.join(MN) + ' | 流水线减右手·绕中心 | 流水线减最好固定 η |\n|---|' + '---:|' * len(M) + '---|---|')
    for g, cs in G2 + [('全部', CL + LE)]:
        pt = {m: agg(m, g, full, balanced) for m in M}
        bs = {m: [] for m in ['rhr', 'sE22']}
        for _ in range(3000):
            draws = {f: [(byfam[f][j], rng.integers(0, 16, 16)) for j in rng.integers(0, len(byfam[f]), len(byfam[f]))] for f in FAM}
            p = agg('pipe', g, draws, balanced)
            for m in bs: bs[m].append(p - agg(m, g, draws, balanced))
        ci = lambda m: f"{pt['pipe'] - pt[m]:+.1f} [{np.percentile(bs[m], 2.5):+.1f}, {np.percentile(bs[m], 97.5):+.1f}]"
        out.append(f'| {g} | ' + ' | '.join(f'{pt[m]:.1f}' for m in M) + f" | {ci('rhr')} | {ci('sE22')} |")
print('\n'.join(out))
