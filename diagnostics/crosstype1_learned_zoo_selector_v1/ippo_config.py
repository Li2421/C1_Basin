"""Freeze inputs for Ring PPO without looking at any controller outcomes."""
from common import *
import pickle


def main():
    configs=[]
    for i,g in enumerate(read(D/'axis2_controller_v1/GAUSSIAN_CONFIGS.json')):
        folder=HERE/'ippo_inputs'/f'ring_ippo{i}'
        folder.mkdir(parents=True,exist_ok=True)
        assert sha(g['manifest'])==g['manifest_sha256']
        manifest=read(g['manifest']);unique={};inputs={g['manifest']:g['manifest_sha256'],g['flow_checkpoint']:g['flow_checkpoint_sha256']}
        for row in manifest['files']:
            if row['split']!='train':continue
            source=Path(g['manifest']).parent/row['file']
            with np.load(source,allow_pickle=False) as z:
                state=json.loads(str(z['initial_state_json'].item()))
            physical={k:state[k] for k in ('positions','velocities','goals')}
            key=json.dumps(physical,sort_keys=True)
            unique[key]=physical;inputs[str(source)]=sha(source)
        keys=sorted(unique);rng=np.random.default_rng(g['validation_seed'])
        order=rng.permutation(len(keys));assert len(keys)>=64
        held=[unique[keys[j]] for j in order[:32]]
        training=[unique[keys[j]] for j in order[32:]]
        assert not ({json.dumps(x,sort_keys=True) for x in held}&{json.dumps(x,sort_keys=True) for x in training})
        write(folder/'TRAINING_STATES.json',training);write(folder/'CHECKPOINT_STATES.json',held)
        old=pickle.loads(Path(g['flow_checkpoint']).read_bytes())['config']
        normal={k:np.asarray(old[k],np.float32).tolist() for k in ('obs_mean','obs_scale','act_mean','act_scale')}
        normal['max_speed']=float(old['max_speed']);write(folder/'NORMALIZATION.json',normal)
        for path in folder.glob('*.json'):inputs[str(path)]=sha(path)
        configs.append(dict(name=f'ring_ippo{i}',family='ring_ippo',locked=True,source_instance=g['name'],
             frame=g['frame'],recipe=g['recipe'],manifest=g['manifest'],manifest_sha256=g['manifest_sha256'],
             seed=202610180400+i,training_states=str(folder/'TRAINING_STATES.json'),
             checkpoint_states=str(folder/'CHECKPOINT_STATES.json'),normalization=str(folder/'NORMALIZATION.json'),
             input_sha256=inputs,environment=manifest['scenario_config'],
             robot_steps=20000000,single_robot_steps=4000000,joint_robot_steps=16000000,
             parallel_envs=32,rollout_steps=125,updates_per_phase=1000,
             checkpoint_every_robot_steps=2000000,checkpoint_selection='original environment nominal-only32 states; largest success count, later tie',
             actor_input='own 23 dimensional G0–G5 chart row only',critic_input='own row only',
             actor_hidden=[256]*3,critic_hidden=[256]*3,activation='tanh',separate_parameters=True,
             optimizer='Adam',learning_rate=3e-4,learning_rate_schedule='linear across 32000 minibatch updates',
             clip=.2,gamma=.99,gae_lambda=.95,entropy=.005,gradient_norm=.5,epochs=4,minibatches=4,
             reward=dict(euclidean_progress=2.5,first_arrival=15,team_success=10,own_collision=-15,other_collision=-5,time=-.01),
             observation_normalization='exact source chart/normalization, then shared running mean/variance frozen in checkpoint',
             single_robot_semantics='one active robot per environment; inactive neighbors omitted from observation and collisions; original goal-stop'))
    write(HERE/'IPPO_TRAINING_CONFIGS_DRAFT.json',configs)
    seal('IPPO_INPUTS.receipt.json',[HERE/'IPPO_TRAINING_CONFIGS_DRAFT.json',*sorted((HERE/'ippo_inputs').rglob('*.json'))],
         no_locked_arm_labels_read=True,training_not_started=True)
    print('Prepared 6 PPO recipes and training/checkpoint state splits')

if __name__=='__main__':main()
