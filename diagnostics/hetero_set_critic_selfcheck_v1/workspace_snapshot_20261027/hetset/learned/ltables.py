"""Tables for learned planners on the frozen validation + test scenes (states 8-23), with the 7 classical controllers
(capability-gate passers) as reference.  Columns: pipeline (frozen 16-proposal choice), best fixed eta sE22,
right-hand rule around centre, right-hand rule goal-relative, safety only, nominal only."""
import os, sys, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
Z = json.load(open(os.path.join(ROOT, 'FINAL_FROZEN.json'))); V = Z['val'] + Z['test']
S = {s['name']: s for s in json.load(open(os.path.join(ROOT, 'scenes_all.json')))}
LEARNED = json.load(open(os.path.join(HERE, 'LEARNED.json')))
R = collections.defaultdict(list)
norm = lambda k: k[5:] if k.startswith('menu:') else k
for f in ['test_menu.jsonl', 'test_props.jsonl', 'test_nominal.jsonl', 'test8.jsonl', 'test_rhr.jsonl', 'ens_val.jsonl', 'fresh.jsonl', 'val8.jsonl', 'rhr_goal.jsonl']:
    for d in map(json.loads, open(os.path.join(ROOT, f))):
        if d['scene'] in V and 8 <= d['state'] <= 23:
            k = d.get('key') or d['action']; R[(d['scene'], norm(k), d['controller'], d['state'])] = d['success']
for d in map(json.loads, open(os.path.join(HERE, 'eval.jsonl'))):
    R[(d['scene'], norm(d['key']), d['controller'], d['state'])] = d['success']
# validation scenes: nominal-only for classical controllers was never run there -> leave as gap
METHODS = ['pipe', 'sE22', 'rhr', 'rhr_goal', 'safety', 'nominal']
NAMES = ['流水线', '最好固定 η', '右手规则·绕中心', '右手规则·按目标方向', '只加安全投影', '不加安全投影']


def key_of(v, m):
    return f"p{Z['choices'][v]['ens16']}" if m == 'pipe' else m


def rate(vs, m, cs):
    x = [R.get((v, key_of(v, m), c, s)) for v in vs for c in cs for s in range(8, 24)]
    x = [y for y in x if y is not None]
    return (100 * np.mean(x), len(x)) if x else (None, 0)


def cls(v):
    f = S[v]['cluster']
    if f == 'box': return '随机起终点·拥挤方形场地'
    if f == 'plus': return '对心对穿·窄十字路口' if 2 * S[v]['params']['w'] <= 0.5 + 1e-9 else '对心对穿·宽十字路口'
    return '对心对穿·环上楔形障碍' if f == 'obst' else '对心对穿·无障碍圆圈'


CLASSICAL = [0, 1, 2, 4, 5, 6, 7]
LIDX = list(range(8, 8 + len(LEARNED)))
LN = [x['name'] for x in LEARNED]
fmt = lambda r: '—' if r[0] is None else f'{r[0]:.1f}'
print('### 20 个验证+测试场景合计（每格 = 场景数 × 16 状态）')
print('| controller | ' + ' | '.join(NAMES) + ' |\n|---|' + '---:|' * len(NAMES))
for name, cs in [(n, [c]) for n, c in zip(LN, LIDX)] + [('**学习型合计**', LIDX), ('7 个经典 controller 合计（参照）', CLASSICAL)]:
    print(f'| {name} | ' + ' | '.join(fmt(rate(V, m, cs)) for m in METHODS) + ' |')
print('\n### 按场景类（学习型合计）')
G = collections.defaultdict(list)
for v in V: G[cls(v)].append(v)
print('| 场景类 | 场景数 | ' + ' | '.join(NAMES) + ' |\n|---|---:|' + '---:|' * len(NAMES))
for k in ['随机起终点·拥挤方形场地', '对心对穿·窄十字路口', '对心对穿·宽十字路口', '对心对穿·环上楔形障碍', '对心对穿·无障碍圆圈']:
    print(f'| {k} | {len(G[k])} | ' + ' | '.join(fmt(rate(G[k], m, LIDX)) for m in METHODS) + ' |')
print('\n### 按场景类（7 个经典 controller，参照）')
print('| 场景类 | 场景数 | ' + ' | '.join(NAMES) + ' |\n|---|---:|' + '---:|' * len(NAMES))
for k in ['随机起终点·拥挤方形场地', '对心对穿·窄十字路口', '对心对穿·宽十字路口', '对心对穿·环上楔形障碍', '对心对穿·无障碍圆圈']:
    print(f'| {k} | {len(G[k])} | ' + ' | '.join(fmt(rate(G[k], m, CLASSICAL)) for m in METHODS) + ' |')
if '--per-scene' in sys.argv:
    print('\n### 每个场景（学习型，逐个 controller）')
    for v in V:
        print(f'\n**{v}**（{cls(v)}）\n| controller | ' + ' | '.join(NAMES) + ' |\n|---|' + '---:|' * len(NAMES))
        for n, c in zip(LN, LIDX):
            print(f'| {n} | ' + ' | '.join(fmt(rate([v], m, [c])) for m in METHODS) + ' |')
