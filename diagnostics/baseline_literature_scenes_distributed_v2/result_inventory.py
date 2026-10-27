"""Resolve exact raw evidence across this experiment and frozen v1 inputs."""
from experiment import *
from copy import deepcopy

def find_raw(rid,stage=None,scene=None):
    if stage:
        path=HERE/stage/'raw'/f'{rid}.json'
        if path.exists():return path
    for manifest in HERE.glob('*_PREFLIGHT.receipt.json'):
        folder=manifest.name.removesuffix('_PREFLIGHT.receipt.json')
        path=HERE/folder/'raw'/f'{rid}.json'
        if path.exists():return path
    scenes=[scene] if scene else list(NAMES)
    for sc in scenes:
        for prefix in ('safety_','fixed_','learning_eval_'):
            path=V1/(prefix+sc)/'raw'/f'{rid}.json'
            if path.exists():
                from source_inputs import verify
                return verify(path)
    raise FileNotFoundError((rid,stage,scene))

def logical(stage):
    sources=[];rows=[]
    for c in read(HERE/f'{stage}_CASES.json'):
        path=find_raw(c['rollout_uid'],stage,c['state']['scene'])
        r=deepcopy(read(path));assert all(r[k]==c[k] for k in ('rollout_uid','state_uid','controller_uid','eta_uid','seed_key'))
        r['case']=c;rows.append(r)
        sources.append(dict(rollout_uid=c['rollout_uid'],path=str(path),sha256=sha(path)))
    return rows,sources

def qualification(scene):
    name='capability_v2_'+scene if scene=='lit_narrow_passage' else 'capability_'+scene
    return read(HERE/f'{name}_QUALIFICATION.json')
