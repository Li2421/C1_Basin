"""Copy learned-planner code, checkpoints, gate/eval records and tables into Basin_C1/diagnostics/hetero_set_learned_planners_v1/."""
import os, sys, json, shutil, glob, csv, collections, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DST = '/home/zhihan/research/Basin_C1/diagnostics/hetero_set_learned_planners_v1'
os.makedirs(DST + '/code', exist_ok=True); os.makedirs(DST + '/checkpoints', exist_ok=True)
for f in ['learned_obs.py', 'learned_models.py', 'learned_actor.py', 'collect_expert.py', 'train_bc.py', 'dagger.py', 'lrun.py', 'ltables.py',
          'chain.sh', 'chain_seed.sh', 'train_three.sh', 'export.py', 'LEARNED.json', 'gate.jsonl', 'eval.jsonl']:
    shutil.copy(os.path.join(HERE, f), os.path.join(DST, 'code' if f.endswith(('.py', '.sh')) else '', f))
for x in json.load(open(os.path.join(HERE, 'LEARNED.json'))):
    d = os.path.join(DST, 'checkpoints', x['name']); os.makedirs(d, exist_ok=True)
    for f in ['meta.json', 'params.msgpack']:
        shutil.copy(os.path.join(HERE, x['ckpt'], f), d)
# DAgger learner outcomes per round (training scenes, safety only)
rounds = {}
for f in sorted(glob.glob(os.path.join(HERE, 'data', 'r*', 'learner_outcomes.json'))):
    rounds[os.path.basename(os.path.dirname(f))] = dict(collections.Counter(o for _, _, o in json.load(open(f))))
expert = dict(collections.Counter(x[2] for x in json.load(open(os.path.join(HERE, 'data', 'r0', 'outcomes.json')))))
json.dump(dict(expert_prioritized_astar_round0=expert, dagger_learner_outcomes=rounds), open(os.path.join(DST, 'TRAINING_ROUNDS.json'), 'w'), indent=1)
out = subprocess.run([sys.executable, os.path.join(HERE, 'ltables.py'), '--per-scene'], capture_output=True, text=True, cwd=ROOT).stdout
open(os.path.join(DST, 'TABLES.md'), 'w').write(out)
print('exported to', DST)
