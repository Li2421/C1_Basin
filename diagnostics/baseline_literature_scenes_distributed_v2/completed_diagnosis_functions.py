"""Exact native diagnostic record collector, restricted to finished source stages."""
from experiment import *
from result_inventory import find_raw,qualification
import collections
import numpy as np

def collect_completed_diagnoses(stages):
    records = []
    sources = []
    for stage in stages:
        for c in read(HERE / f'{stage}_CASES.json'):
            path = find_raw(c['rollout_uid'], stage, c['state']['scene'])
            raw = read(path)
            diag_path = HERE / 'DIAGNOSTICS' / f"{c['rollout_uid']}.json"
            diag = read(diag_path)
            ctl = c['controller_payload']['nominal']
            s = c['state']
            cats = diag['local']
            records.append(dict(layer=c.get('safety_layer', 'central'), scene=s['scene'], N=s['N'], state=s['index'], controller=ctl['controller'], type=ctl['type_code'], method=c['method'], seed=c['seed'], success=raw['success'], upper=min(1, raw['success'] + raw['numerical_failure']), collision=raw['collision'], timeout=raw['timeout'], unknown=raw['numerical_failure'], deadlock_episode=int('robot_filter_equilibrium_deadlock' in cats), wall_episode=int('wall_filter_equilibrium' in cats), deadlock_agents=cats.count('robot_filter_equilibrium_deadlock'), wall_agents=cats.count('wall_filter_equilibrium'), categories=cats, canonical_joint_episode=int('filter_equilibrium_deadlock' in diag.get('canonical_joint', [])), actual_joint_episode=int('filter_equilibrium_deadlock' in diag.get('actual_product_joint', [])), rollout_uid=c['rollout_uid']))
            sources.append(dict(rollout_uid=c['rollout_uid'], logical_stage=stage, raw=str(path), raw_sha256=sha(path), diagnostic=str(diag_path), diagnostic_sha256=sha(diag_path)))
    return (records, sources)

def weighted_rates(rows, extra_metrics=()):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['layer'], r['method']].append(r)
    output = {}
    metrics = ('success', 'upper', 'collision', 'timeout', 'unknown', 'deadlock_episode', 'wall_episode', 'canonical_joint_episode', 'actual_joint_episode', *extra_metrics)
    for (layer, method), records in groups.items():
        clusters = collections.defaultdict(list)
        for r in records:
            clusters[r['type'], r['controller'], r['scene'], r['N'], r['state']].append(r)
        ctls = collections.defaultdict(list)
        for (type_code, ctl, scene, n, index), rr in clusters.items():
            if {r['seed'] for r in rr} != {0, 1}:
                continue
            vector = np.asarray([[*[r[k] for k in metrics], r['deadlock_agents'] / r['N'], r['wall_agents'] / r['N']] for r in rr], float).mean(0)
            ctls[type_code, ctl].append(vector)
        types = collections.defaultdict(list)
        for (type_code, ctl), values in ctls.items():
            types[type_code].append(np.mean(values, axis=0))
        if not types:
            continue
        point = np.mean([np.mean(values, axis=0) for values in types.values()], axis=0)
        output[layer + '|' + method] = dict(rates=dict(zip((*metrics, 'deadlock_agent_fraction', 'wall_agent_fraction'), point.tolist())), controllers=len(ctls), types=sorted(types), scenes=sorted({r['scene'] for r in records}), state_clusters=len(clusters), weighting='Equal type -> controller -> available (scene,N,index) -> paired seeds')
    return output
