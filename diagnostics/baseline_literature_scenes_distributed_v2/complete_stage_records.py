"""Collect frozen completed stages with the sealed main record mapping."""
from experiment import *
from result_inventory import find_raw,qualification

def collect_complete_stages(stages):
    records=[];sources=[]
    for stage in stages:
        for c in read(HERE/f'{stage}_CASES.json'):
            path=find_raw(c['rollout_uid'],stage,c['state']['scene']);raw=read(path)
            diag_path=HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json";diag=read(diag_path)
            ctl=c['controller_payload']['nominal'];s=c['state'];cats=diag['local']
            records.append(dict(layer=c.get('safety_layer','central'),scene=s['scene'],N=s['N'],state=s['index'],controller=ctl['controller'],type=ctl['type_code'],method=c['method'],seed=c['seed'],
                success=raw['success'],upper=min(1,raw['success']+raw['numerical_failure']),collision=raw['collision'],timeout=raw['timeout'],unknown=raw['numerical_failure'],
                deadlock_episode=int('robot_filter_equilibrium_deadlock' in cats),wall_episode=int('wall_filter_equilibrium' in cats),
                deadlock_agents=cats.count('robot_filter_equilibrium_deadlock'),wall_agents=cats.count('wall_filter_equilibrium'),categories=cats,
                canonical_joint_episode=int('filter_equilibrium_deadlock' in diag.get('canonical_joint',[])),actual_joint_episode=int('filter_equilibrium_deadlock' in diag.get('actual_product_joint',[])),rollout_uid=c['rollout_uid']))
            sources.append(dict(rollout_uid=c['rollout_uid'],logical_stage=stage,raw=str(path),raw_sha256=sha(path),diagnostic=str(diag_path),diagnostic_sha256=sha(diag_path)))
    return records,sources

