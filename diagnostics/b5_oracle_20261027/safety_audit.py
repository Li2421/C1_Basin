"""Read-only reconciliation of actual swept collision geometry and original CBF rows."""
import json
import numpy as np
from runtime import OUT,RN,dump
from oracle import inputs
from single_integrator.environment import segment_distance


def main():
    scenes=inputs()[0];configs=[];examples=[]
    rows=list(map(json.loads,(OUT/'candidates.jsonl').open()))
    for name in sorted({r['scene']for r in rows}):
        sc=scenes[name];env=RN.SwapEnv(sc['states'][8],RN.LiteratureConfig(**sc['config']));c=env.config
        physical=env.instance.geometry.margin;cbf=c.agent_radius+c.wall_radius+c.wall_collision_margin+1e-4
        configs.append(dict(scene=name,physical_wall_margin=physical,CBF_wall_margin=cbf,
            physical_minus_CBF_margin=physical-cbf,controller_wall_radius=c.wall_radius,
            geometry_wall_radius=env.instance.geometry.config.wall_radius,wall_count=len(env.walls)))
    for row in rows:
        if not row.get('collision')or not row['trace']or not row['spec'].get('test_label'):continue
        sc=scenes[row['scene']];env=RN.SwapEnv(sc['states'][row['state_index']],RN.LiteratureConfig(**sc['config']));c=env.config
        tr=np.load(row['trace']);p,q=tr['positions'][-2:];u=tr['actions'][-1]
        if not len(env.walls):continue
        distances=segment_distance(p[:,None],q[:,None],env.walls[None,:,0],env.walls[None,:,1])
        i,k=map(int,np.unravel_index(np.argmin(distances),distances.shape));distance=float(distances[i,k])
        geometry=json.loads(str(tr['first_geometry_json']))[-1][i]
        residual=np.asarray(geometry['A'])@u[i]-geometry['lower']
        examples.append(dict(rollout_uid=row['rollout_uid'],trace=row['trace'],scene=row['scene'],
            method=row['spec']['test_label'],state=row['state_index'],controller=row['controller'],seed=row['seed'],
            terminal_step=len(tr['actions']),agent=i,wall=k,wall_segment=env.walls[k],
            before=p[i],after=q[i],executed_velocity=u[i],swept_distance=distance,
            physical_clearance=distance-env.instance.geometry.margin,
            CBF_margin_clearance=distance-(c.agent_radius+c.wall_radius+c.wall_collision_margin+1e-4),
            minimum_recorded_linear_residual=float(residual.min()),
            restoration_steps=row['restoration_steps'],original_constraint_violation_steps=row['constraint_violation_steps']))
    dump(OUT/'safety_geometry_audit.json',dict(configurations=configs,collision_evidence=examples,
        conclusion='Existing bottleneck geometry uses wall_radius=4/600 while LiteratureConfig uses0. Original solver certificates do not certify the larger swept collision geometry. No safety/physics change was made.'))


if __name__=='__main__':main()
