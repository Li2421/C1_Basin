"""Second test batch (scenes_test2.json, seed 2026102792): same eligibility rule and same frozen pipeline as step3.py.
Checks: generator proposals for the old validation/test scenes and the old 16-proposal choices must reproduce
FINAL_FROZEN.json exactly before any choice for the new scenes is accepted."""
import json, pickle, subprocess, shutil, numpy as np, jax
import desc, critic as CR, critic2 as C2
from tab import KEYS
sp = json.load(open('SPLIT.json')); data = pickle.load(open('critic_data_with_test.pkl', 'rb')); TR = sp['train']; VA = sp['val']
Z = json.load(open('FINAL_FROZEN.json')); TE = Z['test']
T2_all = {s['name']: s for s in json.load(open('scenes_test2.json'))}
gate = {}
for l in open('t2_gate.jsonl'):
    d = json.loads(l); t = gate.setdefault(d['scene'], np.full((3, 8, 49), np.nan)); t[d['controller'], d['state'], KEYS.index(d['key'][5:])] = d['success']
T2 = []; per = {}
for n, s in T2_all.items():
    q = gate[n].mean((0, 1)); ok = np.isfinite(gate[n]).all() and .1 <= q.mean() <= .9 and q.max() >= .3
    if ok and per.get(s['cluster'], 0) < 3:
        T2.append(n); per[s['cluster']] = per.get(s['cluster'], 0) + 1
    print(f"candidate {n:30s} mean {q.mean():.2f} best {KEYS[int(q.argmax())]} {q.max():.2f} -> {'TEST2' if n in T2 else 'out'}")
for n in T2_all:
    s = dict(T2_all[n])
    if s['cluster'] == 'obst':
        s['min_opening_over_diameter'] = desc.literature_opening('lit_swap8_obstacles', s['walls'], s['config']['agent_radius'])
    s['features'] = desc.features(s); h = desc.state_h(s); data[n] = dict(t=np.full((3, 24, 49), np.nan), c=desc.descriptor(s), h=h, h_pred=h)
pickle.dump(data, open('critic_data_with_test2.pkl', 'wb'))
json.dump(VA + TE + T2, open('ENS_TARGETS_T2.json', 'w'))
PY = '/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
p = subprocess.run([PY, 'gen_ens_t2.py', 'ENS_TARGETS_T2.json'], capture_output=True, text=True)
P = json.loads([l for l in p.stdout.splitlines() if l.startswith('PROPS')][0][6:])
rep = max(np.abs(np.array(P[v]) - np.array(Z['choices'][v]['proposals'])).max() for v in VA + TE)
print('old val/test proposals reproduce, max abs diff:', rep); assert rep < 1e-6
PT = json.load(open('GEN_ENS_LOSO_TRAIN_PROPOSALS.json')); tp = {n: np.full((3, 24, 16), np.nan) for n in TR}
for l in open('ens_train.jsonl'):
    d = json.loads(l); tp[d['scene']][d['controller'], d['state'], int(d['key'][1:])] = d['success']
nm = CR.norm_of(data, TR)
scenes = {n: dict(E=np.concatenate([CR.MENU, np.array(PT[n])]), t=np.concatenate([data[n]['t'], tp[n]], -1), c=data[n]['c'], h=data[n]['h']) for n in TR}
L = {v: [] for v in TE + T2}
for seed in range(10):
    net, params = C2.train(scenes, nm, seed); ap = jax.jit(lambda p, x: net.apply(p, x))
    for v in TE + T2:
        L[v].append(np.array([[np.asarray(ap(params, C2.Xe(data[v]['c'], data[v]['h'][s], k, nm, np.array(P[v])))) for s in range(24)] for k in range(3)]))
def soft(f): q = np.exp(f - f.max(-1, keepdims=True)); return (q / q.sum(-1, keepdims=True)).mean((0, 1))
choice = {v: int(soft(np.mean(L[v], 0)).argmax()) for v in TE + T2}
mism = [v for v in TE if choice[v] != Z['choices'][v]['ens16']]
print('old test 16-proposal choices reproduce:', len(TE) - len(mism), '/', len(TE), mism); assert not mism
F = {v: dict(proposals=P[v], ens16=choice[v], global_='sE22') for v in T2}
json.dump(dict(test2=T2, choices=F, frozen_before_test2_labels=True, eligibility_rule='mean in [.1,.9], best >= .3, first 3 per cluster'),
          open('FROZEN_T2.json', 'w'), indent=1)
print('frozen test2 scenes', len(T2), T2)
