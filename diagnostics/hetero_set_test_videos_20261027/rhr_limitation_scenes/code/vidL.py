"""Right-hand-rule limitation scenes (crowded boxes + narrow crossroads among the 32 frozen scenes): one video pair per scene,
A = right-hand rule as defined in the literature (push to the right of the robot's own goal direction) fails,
B = pipeline succeeds; same controller, same start state.
usage: vidL.py select | traces OUT | render OUT"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
NAMES = {0: '集中式随机采样模型预测控制', 1: '梯度下降模型预测控制', 2: '分布式模型预测控制（打靶）', 4: '交叉熵方法模型预测控制', 5: '分布式交叉熵方法模型预测控制',
         6: '冲突搜索多机规划', 7: '优先级时空 A* 规划', 8: '多层感知机行为克隆·种子 0', 9: '多层感知机行为克隆·种子 1', 10: '扩散策略·种子 0',
         11: '扩散策略·种子 1', 12: 'Transformer 规划器·种子 0', 13: 'Transformer 规划器·种子 1'}


def select():
    src = open(os.path.join(HERE, 'all32_tables.py')).read(); cut = src.index("G2 = [('经典', CL), ('学习型', LE)]")
    g = {'__file__': os.path.join(HERE, 'all32_tables.py')}; exec(compile(src[:cut], 'all32_head', 'exec'), g)
    S, byfam, r, R, key_of, nice, CH, FN = (g[k] for k in ['S', 'byfam', 'r', 'R', 'key_of', 'nice', 'CH', 'FN'])
    gate = {}
    for f in ['eval.jsonl', 't2_eval.jsonl']:
        for d in map(json.loads, open(os.path.join(HERE, 'gate.jsonl' if f == 'eval.jsonl' else f))):
            if 'robot' in d: gate.setdefault((d['scene'], d['controller']), []).append(d['success'])
    scenes = byfam['box'] + [v for v in byfam['plus'] if 2 * S[v]['params']['w'] <= 0.5 + 1e-9]
    plan = []
    for i, v in enumerate(scenes, 1):
        best = None
        for c in sorted(g['CL'] + g['LE'], key=lambda c: (c < 8, c)):
            if c >= 8 and np.mean(gate[(v, c)]) < 0.95: continue          # learned instance failed the solo gate here
            gain = r([v], 'pipe', [c]) - r([v], 'rhr_goal', [c])
            st = [s for s in range(8, 24) if R[(v, 'rhr_goal', c, s)] == 0 and R[(v, key_of(v, 'pipe'), c, s)] == 1]
            if st and (best is None or gain > best[0] + 1e-9): best = (gain, c, st[0])
        gain, c, s = best; k = CH[v]['ens16']
        plan.append(dict(order=i, scene=v, plain=nice(v), family=FN[S[v]['cluster']] + ('（窄）' if S[v]['cluster'] == 'plus' else ''), controller=c,
                         controller_name=NAMES[c], state=s, pipeline_key=f'p{k}', pipeline_eta=CH[v]['proposals'][k],
                         rhr_goal_rate=round(r([v], 'rhr_goal', [c]), 1), pipeline_rate=round(r([v], 'pipe', [c]), 1)))
        print(i, plan[-1]['family'], nice(v), '|', NAMES[c], '| state', s, '| pipeline', plan[-1]['pipeline_rate'], 'right-hand (goal)', plan[-1]['rhr_goal_rate'])
    json.dump(plan, open(os.path.join(HERE, 'VIDL_PLAN.json'), 'w'), ensure_ascii=False, indent=1)


def run_rhr_record(c, speed=0.35):
    """rhr.run_rhr(c, speed, 'goal') with the trajectory recorded (identical control loop)."""
    import hashlib, jax, runtime_new as RN, rhr
    RULE = rhr.RULE
    env = RN.SwapEnv(c['state'], RN.LiteratureConfig(**c['scenario_payload']['config'])); cfg = env.config; actor = RN.Nominal(c['controller_payload']['nominal'])
    rng_uid = c.get('rng_state_uid', c['state_uid']); root = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026102004), int(hashlib.sha256(rng_uid.encode()).hexdigest()[:8], 16)), c['seed'])
    termination = 'timeout'; hist = {}; pos = [env.positions.copy()]
    for t in range(cfg.max_steps):
        ref = RN.bound(actor.action(env, jax.random.fold_in(root, t)), cfg.max_speed).astype(np.float64); ref[np.linalg.norm(env.goals - env.positions, axis=-1) <= cfg.goal_tolerance] = 0
        snap = env.snapshot(); filters = [RN.SwapSafety(snap, i) for i in range(env.num_agents)]
        v = np.concatenate([f.project(ref[i:i + 1])[0] for i, f in enumerate(filters)])
        a = np.concatenate([f.project(v[i:i + 1])[0] for i, f in enumerate(filters)])
        dist = np.linalg.norm(env.goals - env.positions, axis=1); hist[t] = dist.copy(); hist.pop(t - RULE['steps'] - 1, None)
        active = (hist[t - RULE['steps']] - dist < RULE['threshold']) & (dist > cfg.goal_tolerance) if t >= RULE['steps'] else np.zeros(env.num_agents, bool)
        if speed and active.any():
            target = a.copy(); q = env.positions
            for i in np.flatnonzero(active):
                u = env.goals[i] - q[i]; target[i] += speed * np.array([u[1], -u[0]]) / max(np.linalg.norm(u), 1e-30)
            a = np.concatenate([filters[i].project(target[i:i + 1])[0] if active[i] else a[i:i + 1] for i in range(env.num_agents)])
        _, done, info = env.step(a); termination = info['termination']; pos.append(env.positions.copy())
        if done: break
    return dict(success=int(termination == 'success' and not env.collided), outcome=termination), np.asarray(pos), env.goals.copy()


def job(t):
    import run as R, run8
    p, role, OUT = t; W = run8._W; v = p['scene']; sc = W['scenes'][v]; st = sc['states'][p['state']]
    if role == 'A':
        c = W['make_new'](sc, W['controllers'][p['controller']], st, W['actions'][R.KEYS.index('safety')], 0)
        r, P, G = run_rhr_record(c); eta = None
    else:
        act = dict(key='gen' + p['pipeline_key'], eta_a=list(p['pipeline_eta']), eta_b=list(p['pipeline_eta']), theta=None)
        c = W['make_new'](sc, W['controllers'][p['controller']], st, act, 0); r, a = W['execute'](c); P, G = a['positions'], a['goals']; eta = list(act['eta_a'])
    np.savez_compressed(f"{OUT}/traces/{p['order']:02d}{role}.npz", positions=P, goals=G, walls=np.asarray(sc['walls'], float).reshape(-1, 2, 2), radius=sc['config']['agent_radius'],
                        meta=np.asarray(json.dumps(dict(p, role=role, eta=eta, outcome=r['outcome'], success=int(r['success'])), ensure_ascii=False)))
    return p['order'], role, r['outcome']


def traces(OUT):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp, lrun
    os.makedirs(os.path.join(OUT, 'traces'), exist_ok=True)
    plan = json.load(open(os.path.join(HERE, 'VIDL_PLAN.json')))
    with ProcessPoolExecutor(18, mp_context=mp.get_context('spawn'), initializer=lrun._init) as ex:
        res = sorted(ex.map(job, [(p, role, OUT) for p in plan for role in 'AB']))
    bad = [x for x in res if (x[1] == 'A') == (x[2] == 'success')]
    print('episodes', len(res), 'unexpected outcomes', bad); print(res)


def render_one(f):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Circle
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback']; plt.rcParams['axes.unicode_minus'] = False
    OUT = os.path.dirname(os.path.dirname(f))
    d = np.load(f, allow_pickle=False); m = json.loads(str(d['meta'])); P = d['positions']; G = d['goals']; W = d['walls']; rad = float(d['radius']); T, N, _ = P.shape
    pts = np.concatenate([P.reshape(-1, 2), G] + ([W.reshape(-1, 2)] if len(W) else [])); lo = pts.min(0) - .15; hi = pts.max(0) + .15
    fig, ax = plt.subplots(figsize=(6.6, 7.2), dpi=100); ax.set_aspect('equal'); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_xticks([]); ax.set_yticks([])
    for w in W: ax.plot(w[:, 0], w[:, 1], color='0.25', lw=1.5)
    col = plt.cm.tab20(np.arange(N) % 20); ax.scatter(G[:, 0], G[:, 1], marker='x', c=col, s=40, zorder=2)
    trails = [ax.plot([], [], color=col[i], lw=.8, alpha=.6)[0] for i in range(N)]; disc = [ax.add_patch(Circle(P[0, i], rad, color=col[i], zorder=3)) for i in range(N)]
    how = '右手规则（卡住时往自己前进方向的右侧绕，文献的定义）' if m['role'] == 'A' else '流水线修正：η = (' + ', '.join(f'{x:.2f}' for x in m['eta']) + ')'
    res = {'success': '成功', 'timeout': '超时（卡住）', 'collision': '相撞'}[m['outcome']]
    short = m['plain'].replace('，', '_').replace('（', '_').replace('）', '').replace(' ', '').replace(',', '')
    fam = m['family'].replace('（', '_').replace('）', '')
    name = f"{m['order']:02d}_{fam}_{short}__" + ('A_右手规则_' + ('卡住' if m['outcome'] == 'timeout' else '相撞') if m['role'] == 'A' else 'B_流水线_成功') + '.mp4'
    ttl = ax.set_title('', fontsize=10, loc='left'); wr = FFMpegWriter(fps=20, bitrate=1500); idx = list(range(0, T, 2)) + [T - 1] * 20
    with wr.saving(fig, os.path.join(OUT, name), dpi=100):
        for t in idx:
            for i in range(N): disc[i].center = P[t, i]; trails[i].set_data(P[:t + 1, i, 0], P[:t + 1, i, 1])
            end = f'　→ {res}' if t == T - 1 else ''
            ttl.set_text(f"{m['family']}：{m['plain']}\ncontroller：{m['controller_name']}\n{how}\n第 {t} / {T - 1} 步{end}"); wr.grab_frame()
    plt.close(fig); return name


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'select': select()
    elif mode == 'traces': traces(sys.argv[2])
    else:
        from concurrent.futures import ProcessPoolExecutor
        fs = sorted(glob.glob(os.path.join(sys.argv[2], 'traces', '*.npz')))
        with ProcessPoolExecutor(12) as ex: print('\n'.join(sorted(ex.map(render_one, fs))))
