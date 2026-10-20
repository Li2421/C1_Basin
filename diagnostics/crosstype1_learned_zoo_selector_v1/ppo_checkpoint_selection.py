"""Original-environment, nominal-only checkpoint selection on 32 training-held states."""
from common import *
from design_execution import case
from registry import register,normalized_state
from shared_rollout_db.src.rollout_db import uid,canonical
import copy


def design():
    prereg()
    configs=read(HERE/'IPPO_TRAINING_CONFIGS.json')
    source=read(HERE/'HISTORICAL_BASE_PROTOCOL.json')
    p=dict(states=[],profiles={},types={},eta=source['eta'],native_manifest=source['native_manifest'],
           phase='checkpoint_nominal_training',controller_order=[])
    rows=[];cps={};stateids=set()
    for cfg in configs:
        folder=HERE/'policies'/cfg['name']
        if not (folder/'TRAINING_COMPLETE.receipt.json').exists():continue
        for path,digest in read(folder/'TRAINING_COMPLETE.receipt.json')['files'].items():assert sha(path)==digest
        localstates=[]
        for i,physical in enumerate(read(cfg['checkpoint_states'])):
            token=f"crosstype1_{cfg['name']}_training_held_{i:02}"
            assert all(np.asarray(physical[k]).shape==(4,2) for k in ('positions','velocities','goals'))
            state=dict(uid=token,state_uid=uid('temporary',token),physical=dict(physical,timestep=0),
                 scenario='ring_exchange',scenario_uid=source['profiles'][cfg['source_instance']]['payload']['scenario_uid'],
                 rng_namespace=int(hashlib.sha256(token.encode()).hexdigest()[:8],16),family='crosstype1_training_checkpoint',study_index=i)
            state=normalized_state(state)
            if state['state_uid'] not in stateids:p['states'].append(state);stateids.add(state['state_uid'])
            localstates.append(state)
        for steps in range(2000000,20000001,2000000):
            checkpoint=folder/f'step{steps}.msgpack'
            name=f"{cfg['name']}_step{steps}"
            payload=copy.deepcopy(source['profiles'][cfg['source_instance']]['payload'])
            payload.update(controller_class='ring_ippo',flow_checkpoint_path=str(checkpoint),flow_checkpoint_sha256=sha(checkpoint),
                adapter_manifest_path=cfg['manifest'],observation_frame=cfg['frame'],placement='TT',chain='TT',
                jax_enable_x64=False,conditioning='nominal_TT_project_correct_reproject',
                execution_semantics='crosstype1_ring_ippo_TT_v1',nominal_implementation_sha256=sha(HERE/'zoo_runtime.py'),
                model_implementation_sha256=sha(HERE/'ippo.py'),policy_config_sha256=sha(folder/'CONFIG.json'),
                normalization_sha256=sha(folder/'NORMALIZATION.json'))
            cp=dict(controller_uid=uid('ctl',payload),payload=payload)
            p['profiles'][name]=cp;p['types'][name]='ring_ippo';p['controller_order'].append(name)
            for s in localstates:
                c,armcp=case(s,name,cp,p,'nominal_only',seed=0)
                c.update(instance_name=cfg['name'],training_robot_steps=steps)
                rows.append(c);cps[armcp['controller_uid']]=armcp
    assert rows
    identity=HERE/'EXPERIMENT_REGISTERED_IDENTITY.json'
    if not identity.exists():
        write(identity,dict(experiment_uid=EXP,name='Cross-type learned zoo selector 1',path=str(HERE),
              protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(HERE/'registry.py'),
              metadata_json=canonical(dict(series='crosstype',round=1))))
    register('ppo_checkpoint',p,rows,cps)
    from execution_receipt import freeze
    freeze('ppo_checkpoint',nominal_only=True,training_held_states_only=True,no_locked_correction_labels=True)


def select():
    cases=read(HERE/'ppo_checkpoint_CASES.json');scores={}
    with rollout_db.connect(True) as db:
        for c in cases:
            row=db.execute('SELECT success,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if row is None:raise RuntimeError('Missing PPO checkpoint nominal-only evaluation; do not fabricate unknown')
            name=c['instance_name'];steps=c['training_robot_steps']
            value=scores.setdefault(name,{}).setdefault(steps,dict(success=0,unknown=0,n=0))
            value['success']+=int(row['success'] and not row['numerical_failure']);value['unknown']+=int(row['numerical_failure']);value['n']+=1
    for name,values in scores.items():
        assert len(values)==10 and all(v['n']==32 for v in values.values())
        selected=max(values,key=lambda t:(values[t]['success'],t))
        folder=HERE/'policies'/name;checkpoint=folder/f'step{selected}.msgpack'
        assert not (folder/'CHECKPOINT_SELECTED.receipt.json').exists()
        (folder/'best.msgpack').write_bytes(checkpoint.read_bytes())
        write(folder/'CHECKPOINT_SELECTION.json',dict(selected_robot_steps=selected,source_checkpoint=str(checkpoint),
              tie_rule='later checkpoint',criterion='32 frozen training-held states nominal-only lower success count',
              correction_arm_labels_read=False,scores=values))
        seal(str(folder.relative_to(HERE)/'CHECKPOINT_SELECTED.receipt.json'),
             [folder/'best.msgpack',folder/'CONFIG.json',folder/'NORMALIZATION.json',folder/'CHECKPOINT_SELECTION.json',
              HERE/'ppo_checkpoint_CASES.json',HERE/'ippo.py',HERE/'zoo_runtime.py'],no_locked_correction_labels=True)
    print('Selected training checkpoints for',len(scores),'PPO instances; no correction outcomes opened')

if __name__=='__main__':globals()[sys.argv[1]]()
