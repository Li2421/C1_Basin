"""Current full tables on the frozen 8 validation + 12 test scenes (states 8-23):
7 classical controllers (capability-gate passers) + learned planners (LEARNED.json, only those with complete records).
Methods: pipeline (frozen 16-proposal choice), best fixed eta sE22, right-hand rule around centre, right-hand rule
goal-relative, safety only, nominal only.  usage: final_tables.py [PER_SCENE_OUT.md]"""
import os, sys, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
Z = json.load(open(os.path.join(ROOT, 'FINAL_FROZEN.json'))); VAL, TEST = Z['val'], Z['test']; V = VAL + TEST
S = {s['name']: s for s in json.load(open(os.path.join(ROOT, 'scenes_all.json')))}
LEARNED = json.load(open(os.path.join(HERE, 'LEARNED.json')))
R = {}
norm = lambda k: k[5:] if k.startswith('menu:') else k
for f in ['test_menu.jsonl', 'test_props.jsonl', 'test_nominal.jsonl', 'test8.jsonl', 'test_rhr.jsonl', 'ens_val.jsonl', 'fresh.jsonl', 'val8.jsonl', 'rhr_goal.jsonl', 'val_nominal.jsonl']:
    for d in map(json.loads, open(os.path.join(ROOT, f))):
        if d['scene'] in V and 8 <= d['state'] <= 23:
            R[(d['scene'], norm(d.get('key') or d['action']), d['controller'], d['state'])] = d['success']
for d in map(json.loads, open(os.path.join(HERE, 'eval.jsonl'))):
    R[(d['scene'], norm(d['key']), d['controller'], d['state'])] = d['success']
M = ['pipe', 'sE22', 'rhr', 'rhr_goal', 'safety', 'nominal']
MN = ['流水线', '最好固定 η', '右手规则·绕中心', '右手规则·按目标方向', '只加安全投影', '不加安全投影']
key_of = lambda v, m: f"p{Z['choices'][v]['ens16']}" if m == 'pipe' else m
CL = [(0, '集中式随机采样模型预测控制'), (1, '梯度下降模型预测控制'), (2, '分布式模型预测控制（打靶）'), (4, '交叉熵方法模型预测控制'),
      (5, '分布式交叉熵方法模型预测控制'), (6, '冲突搜索多机规划'), (7, '优先级时空 A* 规划')]
LN = {'mlp': '多层感知机行为克隆', 'diffusion': '扩散策略', 'attention': 'Transformer 规划器'}
LE = []
for i, x in enumerate(LEARNED):
    ci = 8 + i
    if all((v, key_of(v, m), ci, s) in R for v in V for m in M for s in range(8, 24)):
        kind = x['name'].split('_')[1]; seed = 1 if '_s1_' in x['name'] else 0
        LE.append((ci, f'{LN[kind]}·种子 {seed}'))
CI_C, CI_L = [c for c, _ in CL], [c for c, _ in LE]


def rate(vs, m, cs):
    x = [R[(v, key_of(v, m), c, s)] for v in vs for c in cs for s in range(8, 24)]
    return 100 * np.mean(x)


def cls(v):
    f = S[v]['cluster']
    if f == 'box': return '随机起终点·拥挤方形场地'
    if f == 'plus': return '对心对穿·窄十字路口' if 2 * S[v]['params']['w'] <= 0.5 + 1e-9 else '对心对穿·宽十字路口'
    return '对心对穿·环上楔形障碍' if f == 'obst' else '对心对穿·无障碍圆圈'


CLS = ['随机起终点·拥挤方形场地', '对心对穿·窄十字路口', '对心对穿·宽十字路口', '对心对穿·环上楔形障碍', '对心对穿·无障碍圆圈']
hdr = lambda first: f'| {first} | ' + ' | '.join(MN) + ' |\n|---|' + '---:|' * len(MN)
row = lambda name, vs, cs: f'| {name} | ' + ' | '.join(f'{rate(vs, m, cs):.1f}' for m in M) + ' |'
out = []
out.append(f'### 表 1　20 个验证+测试场景合计（每个 controller 一行 = 320 局）\n' + hdr('controller'))
for c, n in CL: out.append(row(n, V, [c]))
out.append(row('**7 个经典 controller 合计**', V, CI_C))
for c, n in LE: out.append(row(n, V, [c]))
out.append(row(f'**{len(LE)} 个学习型实例合计**', V, CI_L))
out.append(row(f'**全部 {len(CI_C) + len(CI_L)} 个合计**', V, CI_C + CI_L))
G = collections.defaultdict(list)
for v in V: G[cls(v)].append(v)
out.append('\n### 表 2　按场景类（左：7 个经典 controller；右：学习型）')
out.append('| 场景类 | 场景数 | 组 | ' + ' | '.join(MN) + ' |\n|---|---:|---|' + '---:|' * len(MN))
for k in CLS:
    for gname, cs in [('经典', CI_C), ('学习型', CI_L)]:
        out.append(f'| {k} | {len(G[k])} | {gname} | ' + ' | '.join(f'{rate(G[k], m, cs):.1f}' for m in M) + ' |')
rng = np.random.default_rng(0)


def arr(m, cs): return np.array([[np.mean([R[(v, key_of(v, m), c, s)] for c in cs]) for s in range(8, 24)] for v in V])


def boot(a, b, B=4000):
    d = a - b; o = []
    for _ in range(B):
        sv = rng.integers(0, len(V), len(V)); ss = rng.integers(0, 16, (len(V), 16)); o.append(d[sv[:, None], ss].mean())
    return f'{100 * d.mean():+.1f} [{100 * np.percentile(o, 2.5):+.1f}, {100 * np.percentile(o, 97.5):+.1f}]'


out.append('\n### 表 3　流水线减各方法（场景再状态配对 bootstrap，95% 区间）\n| 组 | 减最好固定 η | 减右手规则·绕中心 | 减右手规则·按目标方向 | 减只加安全投影 |\n|---|---|---|---|---|')
for gname, cs in [('7 个经典 controller', CI_C), (f'{len(LE)} 个学习型实例', CI_L)]:
    P = arr('pipe', cs)
    out.append(f'| {gname} | ' + ' | '.join(boot(P, arr(m, cs)) for m in ['sE22', 'rhr', 'rhr_goal', 'safety']) + ' |')
print('\n'.join(out))
if len(sys.argv) > 1:
    ps = []
    for v in V:
        ps.append(f"\n#### {v}（{cls(v)}，{'验证' if v in VAL else '测试'}）\n" + hdr('controller'))
        for c, n in CL + LE: ps.append(row(n, [v], [c]))
        ps.append(row('**经典合计**', [v], CI_C)); ps.append(row('**学习型合计**', [v], CI_L))
    open(sys.argv[1], 'w').write('# 逐场景表格（每格 16 局；行 = controller，列 = 方法）\n' + '\n'.join(ps) + '\n')
