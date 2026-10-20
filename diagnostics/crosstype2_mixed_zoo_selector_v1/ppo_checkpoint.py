"""Original nominal-only checkpoint rule, using the original 32 held states."""
from common import *
from zoo_design import case,freeze
from shared_rollout_db.src.rollout_db import uid
import copy


def design():
    prereg();old=D/'crosstype1_learned_zoo_selector_v1'
    original=read(old/'ppo_checkpoint_protocol.json');oldcases=read(old/'ppo_checkpoint_CASES.json')
    states={s['state_uid']:s for s in original['states']};base=read(D/'crosstype1b_generator_critic_pipeline_v1/BASE_PROTOCOL.json')
    p={**base,'states':[],'profiles':{},'controller_order':[],'types':{},'phase':'checkpoint_nominal_training'}
    unique={};rows=[];cps={}
    for cfg in read(HERE/'IPPO_TRAINING_CONFIGS.json'):
        folder=HERE/'policies'/cfg['name'];receipt=read(folder/'TRAINING_COMPLETE.receipt.json')
        for path,digest in receipt['files'].items():assert sha(path)==digest
        oldname=cfg['name'].replace('ring_ippo_conservative','ring_ippo')
        ids=list(dict.fromkeys(c['state_uid'] for c in oldcases if c['instance_name']==oldname));assert len(ids)==32
        localstates=[states[k] for k in ids];unique.update({s['state_uid']:s for s in localstates})
        for steps in range(2000000,20000001,2000000):
            checkpoint=folder/f'step{steps}.msgpack';name=f"{cfg['name']}_step{steps}"
            v=copy.deepcopy(base['profiles'][cfg['source_instance']]['payload'])
            v.update(controller_class='ring_ippo',flow_checkpoint_path=str(checkpoint),flow_checkpoint_sha256=sha(checkpoint),adapter_manifest_path=cfg['manifest'],observation_frame=cfg['frame'],placement='TT',chain='TT',jax_enable_x64=False,conditioning='nominal_TT_project_correct_reproject',execution_semantics='crosstype2_conservative_ippo_TT_v1',nominal_implementation_sha256=sha(HERE/'zoo_runtime.py'),model_implementation_sha256=sha(HERE/'ippo.py'),policy_config_sha256=sha(folder/'CONFIG.json'),normalization_sha256=sha(folder/'NORMALIZATION.json'))
            p['profiles'][name]=dict(controller_uid=uid('ctl',v),payload=v);p['types'][name]='ring_ippo_conservative';p['controller_order'].append(name)
            for s in localstates:
                c,cp=case(p,name,s,16);c.update(instance_name=cfg['name'],training_robot_steps=steps)
                rows.append(c);cps[cp['controller_uid']]=cp
    p['states']=list(unique.values())
    freeze('ppo_checkpoint',p,rows,cps,split='checkpoint_nominal_training',original_32_training_held_states=True,no_correction_arm_outcomes_used=True)


def select():
    cases=read(HERE/'ppo_checkpoint_CASES.json');scores={}
    assert (HERE/'ppo_checkpoint_COMPLETE.receipt.json').exists()
    with rollout_db.connect(True) as db:
        for c in cases:
            row=db.execute('SELECT success,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row is not None
            value=scores.setdefault(c['instance_name'],{}).setdefault(c['training_robot_steps'],dict(success=0,unknown=0,n=0))
            value['success']+=int(row['success'] and not row['numerical_failure']);value['unknown']+=int(row['numerical_failure']);value['n']+=1
    for name,values in scores.items():
        assert len(values)==10 and all(v['n']==32 for v in values.values());step=max(values,key=lambda t:(values[t]['success'],t))
        folder=HERE/'policies'/name;checkpoint=folder/f'step{step}.msgpack';assert not (folder/'CHECKPOINT_SELECTED.receipt.json').exists()
        (folder/'best.msgpack').write_bytes(checkpoint.read_bytes())
        write(folder/'CHECKPOINT_SELECTION.json',dict(utc=utc(),selected_robot_steps=step,scores=values,tie_rule='later checkpoint',criterion='Original 32 training-held states nominal-only success count',correction_labels_used=False))
        seal(str(folder.relative_to(HERE)/'CHECKPOINT_SELECTED.receipt.json'),[folder/'best.msgpack',folder/'CONFIG.json',folder/'NORMALIZATION.json',folder/'CHECKPOINT_SELECTION.json'],original_rule=True)
    print('Selected six conservative checkpoints using original nominal-only rule')


if __name__=='__main__':globals()[sys.argv[1]]()
