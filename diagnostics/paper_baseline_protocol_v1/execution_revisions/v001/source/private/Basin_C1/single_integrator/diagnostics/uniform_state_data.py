"""One perturbed, re-labelled SI transition per source row; no stage weighting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal, segment_distance, point_segment_distance
from single_integrator.expert import Expert
from single_integrator.validate import validate_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--out_dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260908)
    args=p.parse_args()
    metadata,validation=validate_dataset(args.dataset)
    config=Config(**metadata['environment'])
    env=GiveWayEnv(config)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if any(args.out_dir.iterdir()):raise FileExistsError(args.out_dir)
    (args.out_dir/'raw').mkdir()
    rng=np.random.default_rng(args.seed)
    digest=hashlib.sha256();counts={};rejections=0
    for path in sorted((args.dataset/'raw').glob('*.npz')):
        with np.load(path) as source:
            pair,mode=int(source['pair_id']),int(source['mode'])
            if pair>=225:continue
            buffers={k:[] for k in ['observations','actions','next_observations','rewards','dones','wall_collision','agent_collision','stages','next_stages']}
            for t,obs in enumerate(source['observations']):
                for attempt in range(1000):
                    positions=obs[:,:2].astype(float)+rng.uniform(-.02,.02,(2,2))
                    velocity=bounded_nominal(obs[:,2:4]+rng.uniform(-.2,.2,(2,2)),config.max_speed)
                    before=positions-config.dt*velocity
                    wall=segment_distance(before[:,None,:],positions[:,None,:],env.walls[None,:,0,:],env.walls[None,:,1,:])-config.agent_radius-config.wall_radius
                    pair_clearance=point_segment_distance(np.zeros(2),before[0]-before[1],positions[0]-positions[1])-2*config.agent_radius
                    if wall.min()<=config.wall_collision_margin or pair_clearance<=config.agent_collision_margin or env.outside(before).any():
                        rejections+=1;continue
                    try:env.reset(positions)
                    except ValueError:
                        rejections+=1;continue
                    env.velocities=velocity
                    expert=Expert(env,mode);expert.stage=int(source['stages'][t])
                    observation=env.observation();action=expert.action()
                    nxt,reward,done,info=env.step(action)
                    if info['wall_collision'] or info['agent_collision']:
                        rejections+=1;continue
                    break
                else:raise RuntimeError(f'Cannot perturb source row {path}:{t}; no row may be dropped')
                values=dict(observations=observation,actions=action,next_observations=nxt,rewards=reward,
                            dones=bool(info['task_success']),wall_collision=False,agent_collision=False,
                            stages=int(source['stages'][t]),next_stages=expert.stage)
                for key,value in values.items():buffers[key].append(value)
            data={key:np.asarray(value) for key,value in buffers.items()}
            np.testing.assert_array_equal(data['stages'],source['stages'])
            np.testing.assert_allclose(data['next_observations'][:,:,:2],data['observations'][:,:,:2]+config.dt*data['actions'],atol=5e-7,rtol=0)
            np.testing.assert_array_equal(data['next_observations'][:,:,2:4],data['actions'])
            dest=args.out_dir/'raw'/path.name
            np.savez_compressed(dest,**data,pair_id=pair,mode=mode,source_file=path.name,source_step=np.arange(len(data['actions'])),independent_transitions=True,environment_fingerprint=config.fingerprint)
            digest.update(dest.name.encode());digest.update(hashlib.sha256(dest.read_bytes()).digest())
            split='train' if pair<200 else 'val'
            counts[split]=counts.get(split,0)+len(data['actions'])
        if mode==1 and (pair+1)%25==0:print(f'pairs={pair+1} rows={sum(counts.values())} resamples={rejections}',flush=True)
    result=dict(schema='giveway_si_uniform_state_v1',complete=True,environment=config.to_dict(),evaluation_environment=metadata['evaluation_environment'],
                source_dataset=str(args.dataset.resolve()),source_dataset_sha256=validation['dataset_sha256'],dataset_sha256=digest.hexdigest(),
                one_to_one_original_rows=True,stage_weighting=False,recovery_mixture=False,independent_transitions=True,
                position_noise=[-.02,.02],velocity_noise=[-.2,.2],seed=args.seed,counts=counts,resampled_attempts=rejections,
                teacher='same expert, source mode/stage; query action at perturbed state',test_pairs_used=[],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out_dir/'environment.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
