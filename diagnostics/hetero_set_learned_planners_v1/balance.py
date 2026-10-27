"""(1) How similar are the scenes?  Pearson correlation between scenes of (a) the 49-entry eta-menu success profile
(states 0-7, 3 original controllers) and (b) the outcome profile on states 8-23 (every controller x method rate).
(2) Type-balanced tables: random start/goal box scenes and antipodal-swap scenes weighted 50/50
(inside the antipodal type the three families crossroads / ring wedge obstacles / open circle weigh equally).
Stratified bootstrap: scenes resampled within each family, then states within scenes."""
import os, sys, json, collections, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
Z = json.load(open(os.path.join(ROOT, 'FINAL_FROZEN.json'))); VAL, TEST = Z['val'], Z['test']; V = VAL + TEST
S = {s['name']: s for s in json.load(open(os.path.join(ROOT, 'scenes_all.json')))}
LEARNED = json.load(open(os.path.join(HERE, 'LEARNED.json')))
KEYS = ['safety'] + [f'eta{i}' for i in range(16)] + [f'sE{i:02d}' for i in range(32)]
norm = lambda k: k[5:] if k.startswith('menu:') else k
R, P0 = {}, collections.defaultdict(list)
for f in ['test_menu.jsonl', 'test_props.jsonl', 'test_nominal.jsonl', 'test8.jsonl', 'test_rhr.jsonl', 'ens_val.jsonl', 'fresh.jsonl', 'val8.jsonl', 'rhr_goal.jsonl', 'val_nominal.jsonl']:
    for d in map(json.loads, open(os.path.join(ROOT, f))):
        if d['scene'] in V and 8 <= d['state'] <= 23:
            R[(d['scene'], norm(d.get('key') or d['action']), d['controller'], d['state'])] = d['success']
for d in map(json.loads, open(os.path.join(HERE, 'eval.jsonl'))):
    R[(d['scene'], norm(d['key']), d['controller'], d['state'])] = d['success']
for f in ['design.jsonl', 'extra.jsonl', 'test_gate.jsonl']:
    for d in map(json.loads, open(os.path.join(ROOT, f))):
        if d['scene'] in V and d['state'] < 8 and d['controller'] < 3:
            P0[(d['scene'], norm(d.get('key') or d['action']))].append(d['success'])
M = ['pipe', 'sE22', 'rhr', 'rhr_goal', 'safety', 'nominal']
MN = ['流水线', '最好固定 η', '右手规则·绕中心', '右手规则·按目标方向', '只加安全投影', '不加安全投影']
key_of = lambda v, m: f"p{Z['choices'][v]['ens16']}" if m == 'pipe' else m
CL = [0, 1, 2, 4, 5, 6, 7]
LE = [8 + i for i, x in enumerate(LEARNED) if all((v, key_of(v, m), 8 + i, s) in R for v in V for m in M for s in range(8, 24))]
FAM = {'box': '随机起终点·拥挤方形场地', 'plus': '对心对穿·十字路口', 'obst': '对心对穿·环上楔形障碍', 'swap': '对心对穿·无障碍圆圈'}
fam = lambda v: S[v]['cluster']
byfam = collections.defaultdict(list)
for v in V: byfam[fam(v)].append(v)

# ---------- (1) similarity
prof = {v: np.array([np.mean(P0[(v, k)]) for k in KEYS]) for v in V}
outp = {v: np.array([np.mean([R[(v, key_of(v, m), c, s)] for s in range(8, 24)]) for c in CL + LE for m in M[:-1]]) for v in V}


def corr_table(P, title):
    C = {(a, b): np.corrcoef(P[a], P[b])[0, 1] for a in V for b in V if a != b}
    F = ['box', 'plus', 'obst', 'swap']
    print(f'\n#### {title}：场景两两相关系数的平均（同族不含自身）')
    print('| | ' + ' | '.join(FAM[f] for f in F) + ' |\n|---|' + '---:|' * 4)
    for f in F:
        print(f'| {FAM[f]} | ' + ' | '.join(f"{np.mean([C[(a, b)] for a in byfam[f] for b in byfam[g] if a != b]):.2f}" for g in F) + ' |')
    anti = byfam['plus'] + byfam['obst'] + byfam['swap']
    w_anti_cross = np.mean([C[(a, b)] for a in anti for b in anti if a != b and fam(a) != fam(b)])
    w_box = np.mean([C[(a, b)] for a in byfam['box'] for b in byfam['box'] if a != b])
    bt = np.mean([C[(a, b)] for a in byfam['box'] for b in anti])
    print(f'对心对穿三族之间（不同族）平均 {w_anti_cross:.2f}；方形场地内部 {w_box:.2f}；方形场地与对心对穿之间 {bt:.2f}')


corr_table(prof, '(a) 49 个菜单 η 的成功率曲线（状态 0-7，3 个原 controller）')
corr_table(outp, '(b) 状态 8-23 上"controller × 方法"的成功率向量')


# ---------- (2) type-balanced tables
def bal(m, cs, pick=None):
    """50% box + 50% antipodal (three families equal).  pick: optional resampled scene lists per family + state indices."""
    def fmean(f):
        vs = byfam[f] if pick is None else pick[f]
        return np.mean([np.mean([R[(v, key_of(v, m), c, s)] for c in cs for s in (range(8, 24) if pick is None else pick['st'][v])]) for v in vs])
    return 100 * (0.5 * fmean('box') + 0.5 * np.mean([fmean(f) for f in ['plus', 'obst', 'swap']]))


NAMES = {0: '集中式随机采样模型预测控制', 1: '梯度下降模型预测控制', 2: '分布式模型预测控制（打靶）', 4: '交叉熵方法模型预测控制', 5: '分布式交叉熵方法模型预测控制',
         6: '冲突搜索多机规划', 7: '优先级时空 A* 规划'}
LN = {'mlp': '多层感知机行为克隆', 'diffusion': '扩散策略', 'attention': 'Transformer 规划器'}
for i, x in enumerate(LEARNED):
    k = x['name'].split('_')[1]; NAMES[8 + i] = f"{LN[k]}·种子 {1 if '_s1_' in x['name'] else 0}"
print('\n#### 表 1′　两大类等权（方形场地 50% + 对心对穿 50%，对心三族内部等权）\n| controller | ' + ' | '.join(MN) + ' |\n|---|' + '---:|' * len(M))
for c in CL: print(f'| {NAMES[c]} | ' + ' | '.join(f'{bal(m, [c]):.1f}' for m in M) + ' |')
print(f'| **7 个经典 controller 合计** | ' + ' | '.join(f'{bal(m, CL):.1f}' for m in M) + ' |')
for c in LE: print(f'| {NAMES[c]} | ' + ' | '.join(f'{bal(m, [c]):.1f}' for m in M) + ' |')
print(f'| **{len(LE)} 个学习型实例合计** | ' + ' | '.join(f'{bal(m, LE):.1f}' for m in M) + ' |')
print(f'| **全部 {len(CL) + len(LE)} 个合计** | ' + ' | '.join(f'{bal(m, CL + LE):.1f}' for m in M) + ' |')
rng = np.random.default_rng(0)
print('\n#### 表 3′　两大类等权下，流水线减各方法（分层 bootstrap：族内重抽场景，再重抽状态；95% 区间）\n| 组 | 减最好固定 η | 减右手·绕中心 | 减右手·按目标方向 | 减只加安全投影 |\n|---|---|---|---|---|')
for gname, cs in [('7 个经典 controller', CL), (f'{len(LE)} 个学习型实例', LE), (f'全部 {len(CL) + len(LE)} 个', CL + LE)]:
    point = {m: bal('pipe', cs) - bal(m, cs) for m in ['sE22', 'rhr', 'rhr_goal', 'safety']}
    # precompute per-scene per-state means for speed
    A = {m: {v: np.array([np.mean([R[(v, key_of(v, m), c, s)] for c in cs]) for s in range(8, 24)]) for v in V} for m in M[:-1]}
    boots = {m: [] for m in point}
    for _ in range(3000):
        pick = {f: [byfam[f][j] for j in rng.integers(0, len(byfam[f]), len(byfam[f]))] for f in byfam}
        draws = {f: [(v, rng.integers(0, 16, 16)) for v in pick[f]] for f in pick}   # same resampled states for every method
        def fmean(m, f): return np.mean([A[m][v][ss].mean() for v, ss in draws[f]])
        def b(m): return 100 * (0.5 * fmean(m, 'box') + 0.5 * np.mean([fmean(m, f) for f in ['plus', 'obst', 'swap']]))
        bp = b('pipe')
        for m in boots: boots[m].append(bp - b(m))
    print(f'| {gname} | ' + ' | '.join(f'{point[m]:+.1f} [{np.percentile(boots[m], 2.5):+.1f}, {np.percentile(boots[m], 97.5):+.1f}]' for m in ['sE22', 'rhr', 'rhr_goal', 'safety']) + ' |')
